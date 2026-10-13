"""Vision models whose weights get generated. Every model ends in a Linear named `fc`."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CLASSES = 10
HEAD = ("fc.weight", "fc.bias")


class TinaCNN(nn.Module):
    """~5.4K parameters, the size of Tina's CNN; every parameter is generated."""

    def __init__(self, num_classes=MAX_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, widths, blocks, num_classes=MAX_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(F.relu(self.bn1(self.conv1(x))))
        return self.fc(x.mean((2, 3)))


class ClipHead(nn.Module):
    """Classifier head over frozen CLIP ViT-B/32 image features (Tina's ViT setting)."""

    def __init__(self, num_classes=MAX_CLASSES, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)


ARCHS = {
    "clip_head": lambda k: ClipHead(k),                                   # ~5.1K params on frozen CLIP
    "tina_cnn": lambda k: TinaCNN(k),                                     # ~5.4K params
    "resnet20": lambda k: ResNet((16, 32, 64), (3, 3, 3), k),             # ~0.27M params
    "resnet18": lambda k: ResNet((64, 128, 256, 512), (2, 2, 2, 2), k),   # ~11M params
}


def uses_features(arch):
    return arch == "clip_head"


def build(arch, num_classes=MAX_CLASSES):
    return ARCHS[arch](num_classes)


def param_spec(model):
    """(name, shape) of every floating-point tensor in the state dict (weights + BN statistics)."""
    return [(k, tuple(v.shape)) for k, v in model.state_dict().items() if v.dtype.is_floating_point]


def flatten(model):
    sd = model.state_dict()
    return torch.cat([sd[k].detach().float().reshape(-1).cpu() for k, _ in param_spec(model)])


def load_flat(model, flat):
    """Load a flat parameter vector (as produced by `flatten` or the generator) into `model`."""
    sd, i = {}, 0
    for name, shape in param_spec(model):
        n = int(torch.Size(shape).numel())
        t = flat[i:i + n].view(shape)
        if name.endswith("running_var"):
            t = t.clamp_min(1e-5)
        sd[name] = t
        i += n
    model.load_state_dict(sd, strict=False)
    return model
