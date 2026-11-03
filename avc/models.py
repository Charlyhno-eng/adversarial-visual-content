"""Two public visual representations, distinct from softmax outputs."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (resnet18, ResNet18_Weights,
                                mobilenet_v3_small, MobileNet_V3_Small_Weights)
from torchvision.transforms import functional as TF


class VisionModels:
    def __init__(self, cache: Path, device='cpu'):
        torch.hub.set_dir(str(cache))
        self.device = torch.device(device)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        mobile = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.nets = {
            'resnet18': torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.Flatten()),
            'mobilenet_v3_small': torch.nn.Sequential(mobile.features, mobile.avgpool, torch.nn.Flatten()),
        }
        for net in self.nets.values():
            net.to(self.device).eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def tensor(self, im):
        return TF.to_tensor(im).unsqueeze(0).to(self.device)

    def features(self, x):
        x = TF.center_crop(TF.resize(x, 256, antialias=True), [224, 224])
        x = (x - self.mean) / self.std
        return {name: F.normalize(net(x), dim=1) for name, net in self.nets.items()}

    @torch.no_grad()
    def compare(self, reference_features, im):
        values = self.features(self.tensor(im))
        return {f'{name}_cosine': float((reference_features[name] * vec).sum())
                for name, vec in values.items()}


def as_image(x):
    from PIL import Image
    pixels = x.detach().clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)
