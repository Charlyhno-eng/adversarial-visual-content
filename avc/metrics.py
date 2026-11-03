"""Measurements on exported pixels, distinguishing prediction from identity."""
from io import BytesIO
import imagehash
import numpy as np
import pdqhash
from PIL import Image
from skimage.metrics import structural_similarity

PDQ_THRESHOLD = 31  # Public recommendation; not an Instagram/TikTok threshold.
HASH_THRESHOLD = 10  # Configurable convention for this benchmark.


def hash_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    ref = np.array(reference.convert('RGB'))
    cand = np.array(candidate.convert('RGB'))
    rh, rq = pdqhash.compute(ref)
    ch, cq = pdqhash.compute(cand)
    ph = int(imagehash.phash(reference) - imagehash.phash(candidate))
    dh = int(imagehash.dhash(reference) - imagehash.dhash(candidate))
    distance = int(np.count_nonzero(rh != ch))
    valid = min(int(rq), int(cq)) >= 50
    # Also test the eight rotations/reflections, without claiming to reproduce a platform.
    variants, _ = pdqhash.compute_dihedral(cand)
    dihedral = min(int(np.count_nonzero(rh != h)) for h in variants)
    return {'phash_distance_64': ph, 'dhash_distance_64': dh,
            'pdq_distance_256': distance, 'pdq_dihedral_distance_256': dihedral,
            'pdq_reference_quality': int(rq), 'pdq_candidate_quality': int(cq),
            'pdq_valid': valid}


def visual_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    # Only for tests with different sizes, resize the candidate to match the reference.
    resized = candidate.size != reference.size
    if resized:
        candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
    a = np.asarray(reference.convert('RGB'))
    b = np.asarray(candidate.convert('RGB'))
    delta = a.astype(float) - b.astype(float)
    mse = float(np.mean(delta ** 2))
    return {'ssim': float(structural_similarity(a, b, channel_axis=2, data_range=255)),
            'mae_levels': float(np.abs(delta).mean()),
            'linf_levels': int(np.abs(delta).max()),
            'psnr_db': None if mse == 0 else float(10 * np.log10(255**2 / mse)),
            'visual_comparison_resized': resized}


def transformed(image: Image.Image, kind: str) -> Image.Image:
    if kind == 'png':
        return image.copy()
    if kind.startswith('jpeg'):
        q = int(kind.removeprefix('jpeg'))
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    if kind == 'resize75':
        return image.resize(tuple(max(8, round(n * .75)) for n in image.size), Image.Resampling.LANCZOS)
    raise ValueError(f'Unknown transformation: {kind}')


def local_decision(row: dict, hash_threshold=HASH_THRESHOLD, cosine_threshold=.8) -> dict:
    tests = {
        'phash_separated': row['phash_distance_64'] > hash_threshold,
        'dhash_separated': row['dhash_distance_64'] > hash_threshold,
        'pdq_separated': row['pdq_valid'] and row['pdq_dihedral_distance_256'] > PDQ_THRESHOLD,
        'resnet_separated': row['resnet18_cosine'] < cosine_threshold,
        'mobilenet_separated': row['mobilenet_v3_small_cosine'] < cosine_threshold,
    }
    return {**tests, 'local_tests_passed': sum(bool(x) for x in tests.values()),
            'all_local_detectors_separated': all(tests.values())}
