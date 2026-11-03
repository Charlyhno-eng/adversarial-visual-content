from PIL import Image
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from avc.metrics import hash_metrics, visual_metrics, local_decision, transformed
from avc.optimize import generate
from avc.__main__ import load


@pytest.fixture
def photo():
    rng = np.random.default_rng(12)
    return Image.fromarray(rng.integers(0, 256, (40, 32, 3), dtype=np.uint8))


def test_identical_export_has_zero_hash_distance_and_unit_ssim(photo, tmp_path):
    path = tmp_path / 'image.png'
    photo.save(path)
    reread = Image.open(path).convert('RGB')
    hashes = hash_metrics(photo, reread)
    assert hashes['phash_distance_64'] == 0
    assert hashes['dhash_distance_64'] == 0
    assert hashes['pdq_distance_256'] == 0
    assert hashes['pdq_dihedral_distance_256'] == 0
    visual = visual_metrics(photo, reread)
    assert visual['ssim'] == 1
    assert visual['linf_levels'] == 0
    assert visual['psnr_db'] is None


def test_low_pdq_quality_cannot_be_reported_as_separated():
    row = {'phash_distance_64': 20, 'dhash_distance_64': 20,
           'pdq_dihedral_distance_256': 128, 'pdq_valid': False,
           'resnet18_cosine': .3, 'mobilenet_v3_small_cosine': .3}
    decision = local_decision(row)
    assert decision['local_tests_passed'] == 4
    assert not decision['all_local_detectors_separated']


def test_threshold_boundaries_and_dihedral_match():
    row = {'phash_distance_64': 10, 'dhash_distance_64': 11,
           'pdq_dihedral_distance_256': 31, 'pdq_valid': True,
           'resnet18_cosine': .8, 'mobilenet_v3_small_cosine': .79}
    result = local_decision(row)
    assert not result['phash_separated']
    assert not result['pdq_separated']
    assert not result['resnet_separated']
    assert result['local_tests_passed'] == 2


def test_transforms_and_small_input_validation(photo, tmp_path):
    assert transformed(photo, 'jpeg75').size == photo.size
    assert transformed(photo, 'resize75').size == (24, 30)
    with pytest.raises(ValueError):
        transformed(photo, 'unsupported')
    tiny = tmp_path / 'tiny.png'
    Image.new('RGB', (7, 10)).save(tiny)
    with pytest.raises(ValueError, match='too small'):
        load(tiny, 512)


def test_projected_attack_obeys_exported_pixel_budget(photo):
    # Lightweight local model: test projection without downloading weights.
    class Models:
        def tensor(self, image):
            return torch.tensor(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        def features(self, x):
            return {'test': F.normalize(F.adaptive_avg_pool2d(x, (4, 4)).flatten(1), dim=1)}
    for budget, strategy, candidate in generate(photo, Models(), [2], steps=2):
        delta = np.array(candidate).astype(int) - np.array(photo).astype(int)
        assert abs(delta).max() <= budget
        assert candidate.size == photo.size


def test_selection_prioritizes_duplicate_signatures_and_respects_quality():
    from avc.__main__ import choose_candidate
    def candidate(name, ssim, hashes, all_separated=False):
        return {'candidate': name, 'ssim': ssim, 'worst_hash_tests_passed': hashes,
                'worst_tests_passed': hashes, 'mean_tests_passed': float(hashes),
                'all_local_detectors_separated': all_separated}
    choices = [candidate('features', .995, 0), candidate('hashes', .985, 2),
               candidate('too_visible', .94, 3, True)]
    result = choose_candidate(choices, .98)
    assert result['candidate'] == 'hashes'
    assert not result['all_local_detectors_separated']
    # Successful distances must not hide a failed quality constraint.
    result = choose_candidate([choices[-1]], .98)
    assert not result['quality_constraint_met']
    assert not result['all_local_detectors_separated']
