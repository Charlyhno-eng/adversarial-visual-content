"""uv run python -m avc generate --input data/source_rts.image --output outputs/multi"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision
from .metrics import hash_metrics, visual_metrics, transformed, local_decision
from .models import VisionModels
from .optimize import generate

TRANSFORMS = ('png', 'jpeg90', 'jpeg75', 'resize75')


def load(path, max_side):
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if min(image.size) < 8:
        raise ValueError('Image too small: minimum 8 pixels per side.')
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def evaluate(reference, candidate, models, clean_features, name, args):
    rows = []
    for transform in TRANSFORMS:
        image = transformed(candidate, transform)
        row = {'candidate': name, 'transform': transform,
               **hash_metrics(reference, image), **visual_metrics(reference, image),
               **models.compare(clean_features, image)}
        row.update(local_decision(row, args.hash_threshold, args.cosine_threshold))
        rows.append(row)
    return rows


def choose_candidate(summaries, min_ssim):
    eligible = [s for s in summaries if s['ssim'] >= min_ssim]
    chosen = max(eligible or summaries, key=lambda s: (
        s['worst_hash_tests_passed'], s['worst_tests_passed'], s['mean_tests_passed'], s['ssim']))
    return {**chosen, 'quality_constraint_met': bool(eligible),
            'all_local_detectors_separated': bool(eligible) and chosen['all_local_detectors_separated'],
            'platform_result': 'not tested'}


def write_report(out, config, table, selection):
    table.to_csv(out / 'measurements.csv', index=False)
    payload = {'config': config, 'selection': selection,
               'measurements': json.loads(table.to_json(orient='records'))}
    (out / 'report.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    selected = selection['candidate']
    text = f'''# Local evaluation of visual signatures

Selected candidate: **{selected}**. PNG SSIM: **{selection['ssim']:.5f}**.

**All local detectors separated in every test: {selection['all_local_detectors_separated']}**.
If this result is false, the benchmark objective is not achieved. A large hash distance or low feature similarity does not mean “a new image for Instagram/TikTok.”

## Protocol

- Working reference: {config['working_size']}, source preserved separately.
- Budgets: {config['budgets']} levels out of 255; {config['steps']} iterations per strategy.
- Minimum PNG quality: SSIM ≥ {config['min_ssim']}. This does not certify invisibility.
- Strategies: approximate DCT signatures on pixels or smooth fields, and signatures + ResNet-18/MobileNet representations.
- Optimization with rounding, JPEG 80/90 (approximate gradient), and resizing to 80%.
- Actual evaluation after PNG export, JPEG 90, JPEG 75, and resizing to 75%. The reference signature remains that of the initial clean image for every test.
- PDQ: separation if distance > 31, quality ≥ 50, including the closest match across the candidate's eight orientations.
- pHash/dHash: separation if distance > {config['hash_threshold']} bits out of 64, a local convention.
- Two deep features: separation if cosine < {config['cosine_threshold']}, a local convention; these are not probabilities.
- All criteria must pass simultaneously across the four transformations for local success. Selection prioritizes candidates meeting the quality constraint, then the minimum number of separated perceptual signatures across transformations, then the minimum and mean number of separated detectors, and finally SSIM.

Instagram's and TikTok's models and settings, content indexes, and recommendation signals are not accessible here. No upload was performed, and no bypass of these platforms was validated. Audio, temporal sequences, OCR, and metadata are not tested. This system processes still images, not videos.

See `measurements.csv` for all measurements and `report.json` for parameters, versions, and the source SHA-256.

## Sources

- [PDQ: recommended distances and quality](https://github.com/facebook/ThreatExchange/blob/main/pdq/README.md).
- [PDQ bindings used for actual evaluation](https://pypi.org/project/pdqhash/).
- [ImageHash: pHash and dHash](https://github.com/JohannesBuchner/imagehash).
- [Torchvision: pretrained models](https://docs.pytorch.org/vision/stable/models.html).

The DCT surrogate is an approximation used to guide optimization, not a certified differentiable reproduction of PDQ. PDQ measurements come exclusively from `pdqhash` on exported pixels. No protection against training is demonstrated.
'''
    (out / 'report.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description='Local generation and measurement, with no guarantee for any private platform.')
    parser.add_argument('command', choices=['generate', 'evaluate'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help='Image to measure with evaluate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--max-side', type=int, default=512, help='0 preserves native resolution')
    parser.add_argument('--min-ssim', type=float, default=.98)
    parser.add_argument('--hash-threshold', type=int, default=10)
    parser.add_argument('--cosine-threshold', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps < 1 or any(b < 1 or b > 32 for b in args.budgets):
        parser.error('Steps ≥ 1 and budgets between 1 and 32 are required.')
    if args.max_side != 0 and args.max_side < 8:
        parser.error('--max-side must be 0 or ≥ 8.')
    if not 0 < args.min_ssim <= 1 or not -1 <= args.cosine_threshold <= 1 or not 0 <= args.hash_threshold <= 64:
        parser.error('Thresholds out of range.')
    if args.command == 'evaluate' and args.candidate is None:
        parser.error('evaluate requires --candidate.')
    out = args.output
    if out.exists() and any(out.iterdir()):
        parser.error('The output directory must be empty to preserve existing experiments.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    reference = load(args.input, args.max_side)
    reference.save(out / 'reference.png')
    models = VisionModels(Path.cwd() / '.cache' / 'torch', args.device)
    with torch.no_grad():
        clean = models.features(models.tensor(reference))
    rows = evaluate(reference, reference, models, clean, 'identity', args)
    summaries = []

    def record(image, name, budget=None):
        path = out / f'{name}.png'
        image.save(path)
        # All measurements use the reloaded file, not the floating-point tensor.
        image = Image.open(path).convert('RGB')
        measured = evaluate(reference, image, models, clean, name, args)
        if budget is not None:
            assert measured[0]['linf_levels'] <= budget
        rows.extend(measured)
        ssim = measured[0]['ssim']
        summary = {'candidate': name, 'budget': budget, 'ssim': ssim,
                   'quality_eligible': ssim >= args.min_ssim,
                   'worst_hash_tests_passed': min(sum(bool(r[k]) for k in ('phash_separated', 'dhash_separated', 'pdq_separated')) for r in measured),
                   'worst_tests_passed': min(r['local_tests_passed'] for r in measured),
                   'mean_tests_passed': sum(r['local_tests_passed'] for r in measured) / len(measured),
                   'all_local_detectors_separated': all(r['all_local_detectors_separated'] for r in measured)}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    if args.command == 'generate':
        for budget, strategy, image in generate(reference, models, args.budgets, args.steps, args.seed,
                                               callback=lambda s: print(s, flush=True)):
            record(image, f'{strategy}_{budget:02d}', budget)
    else:
        candidate = ImageOps.exif_transpose(Image.open(args.candidate)).convert('RGB')
        record(candidate, 'external_candidate')
    selection = choose_candidate(summaries, args.min_ssim)
    chosen = selection
    Image.open(out / f'{chosen["candidate"]}.png').save(out / 'selection.png')
    pd.DataFrame(summaries).to_csv(out / 'candidates.csv', index=False)
    config = {'input': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'working_size': reference.size, 'command': args.command,
              'budgets': args.budgets, 'steps': args.steps, 'seed': args.seed,
              'min_ssim': args.min_ssim, 'hash_threshold': args.hash_threshold,
              'cosine_threshold': args.cosine_threshold, 'max_side': args.max_side,
              'device': args.device, 'python': platform.python_version(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    write_report(out, config, pd.DataFrame(rows), selection)
    print('Report:', out / 'report.md', flush=True)


if __name__ == '__main__':
    main()
