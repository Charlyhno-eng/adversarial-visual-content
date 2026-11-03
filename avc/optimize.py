"""Projected gradients: approximate DCT signatures + deep representations.

The DCT surrogate is not PDQ. Only the pdqhash evaluation is authoritative.
"""
import math
import torch
import torch.nn.functional as F
from .models import as_image
from .metrics import transformed


def dct_matrix(size, count, start, device):
    k = torch.arange(start, start + count, device=device).float()[:, None]
    j = torch.arange(size, device=device).float()[None, :]
    return math.sqrt(2 / size) * torch.cos(math.pi / size * (j + .5) * k)


class HashSurrogate:
    def __init__(self, x):
        self.matrices = [(32, dct_matrix(32, 8, 0, x.device)),
                         (64, dct_matrix(64, 16, 1, x.device))]
        self.targets = []
        for index, coeff in enumerate(self.coefficients(x)):
            centered = coeff if index == 2 else coeff - coeff.median()
            # Target the bits closest to their decision boundary rather than all bits.
            count = 20 if coeff.numel() == 64 else 64
            ids = centered.abs().argsort()[:count]
            signs = torch.where(centered[ids] >= 0, 1., -1.)
            scale = coeff.std().clamp_min(.01).detach()
            self.targets.append((ids, signs.detach(), scale))

    def coefficients(self, x):
        gray = (x * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        values = []
        for size, matrix in self.matrices:
            small = F.interpolate(gray, size=(size, size), mode='bilinear', align_corners=False, antialias=True)[0, 0]
            values.append((matrix @ small @ matrix.T).flatten())
        # dHash surrogate: differences between columns of a luminance thumbnail.
        tiny = F.interpolate(gray, size=(8, 9), mode='bilinear', align_corners=False, antialias=True)
        values.append((tiny[..., 1:] - tiny[..., :-1]).flatten())
        return values

    def loss(self, x):
        losses = []
        for index, (c, (ids, signs, scale)) in enumerate(zip(self.coefficients(x), self.targets)):
            centered = c if index == 2 else c - c.median()
            losses.append(F.relu(.025 + signs * centered[ids] / scale).mean())
        return sum(losses)


def eot(x, step):
    """Actual quantized/JPEG forward pass; approximate identity derivative (STE).

    Evaluation transformations remain independent and nondifferentiable.
    """
    mode = step % 4
    if mode == 1:
        h, w = x.shape[-2:]
        return F.interpolate(x, size=(max(8, round(h * .8)), max(8, round(w * .8))),
                             mode='bilinear', align_corners=False, antialias=True)
    image = as_image(x)
    if mode in (2, 3):
        image = transformed(image, 'jpeg80' if mode == 2 else 'jpeg90')
    from torchvision.transforms import functional as TF
    exact = TF.to_tensor(image).unsqueeze(0).to(x.device)
    return x + (exact - x).detach()


def generate(reference, models, budgets, steps, seed=17, callback=None):
    torch.manual_seed(seed)
    x0 = models.tensor(reference)
    with torch.no_grad():
        clean = models.features(x0)
    surrogate = HashSurrogate(x0)
    for budget in budgets:
        eps = budget / 255
        for strategy in ('hash', 'ensemble', 'smooth'):
            # Nonzero initialization: the cosine derivative is zero at identity.
            x = (x0 + torch.empty_like(x0).uniform_(-eps / 4, eps / 4)).clamp(0, 1)
            small_h = max(8, round(48 * x0.shape[-2] / max(x0.shape[-2:])))
            small_w = max(8, round(48 * x0.shape[-1] / max(x0.shape[-2:])))
            field = torch.zeros((1, 1, small_h, small_w), device=x0.device)
            for step in range(steps):
                if strategy == 'smooth':
                    field = field.detach().requires_grad_(True)
                    delta = F.interpolate(field, size=x0.shape[-2:], mode='bilinear', align_corners=False)
                    x = (x0 + delta).clamp(0, 1)
                else:
                    x = x.detach().requires_grad_(True)
                view = eot(x, step)
                loss = surrogate.loss(view) * (4 if strategy == 'hash' else 1)
                if strategy == 'ensemble':
                    feats = models.features(view)
                    loss = loss + sum((feats[name] * clean[name]).sum() for name in feats)
                variable = field if strategy == 'smooth' else x
                gradient, = torch.autograd.grad(loss, variable)
                with torch.no_grad():
                    if strategy == 'smooth':
                        field = (field - (.5 / 255) * gradient.sign()).clamp(-eps, eps)
                        delta = F.interpolate(field, size=x0.shape[-2:], mode='bilinear', align_corners=False)
                        x = (x0 + delta).clamp(0, 1)
                    else:
                        x -= (.5 / 255) * gradient.sign()
                        x = torch.maximum(torch.minimum(x, x0 + eps), x0 - eps).clamp(0, 1)
                if callback and (step + 1) % 40 == 0:
                    callback(f'Budget {budget}/255, strategy {strategy}, step {step+1}/{steps}')
            yield budget, strategy, as_image(x)
