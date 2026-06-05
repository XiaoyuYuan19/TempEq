"""ResNet-18 encoder + projection head for TempEq SSL training.

Only the encoder is kept after pretraining; the projection head is
discarded. Linear probe uses the frozen encoder.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class Encoder(nn.Module):
    """ResNet-18 backbone (512-D features, fc stripped)."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

    def forward(self, x):
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """Configurable MLP projection head.

    n_layers=2 -> SimCLR original (Linear -> ReLU -> Linear)
    n_layers=3 -> VICReg / BarlowTwins style
    use_bn     -> BN1d after every Linear except the last
    """

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128,
                 n_layers=2, use_bn=False):
        super().__init__()
        assert n_layers >= 2
        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SSLModel(nn.Module):
    """Encoder + projection head wrapper used during SSL training.

    Frame mode: input x has shape (B, 3, H, W); output (B, proj_dim).

    Clip mode: input x has shape (B, T, 3, H, W); encoder is applied
    per-frame and mean-pooled over T before the projector. Output is
    still (B, proj_dim) -- the same downstream API as frame mode.
    """

    def __init__(self, proj_dim=128, proj_hidden=512, proj_layers=2,
                 proj_bn=False):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead(
            in_dim=self.encoder.feature_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_dim,
            n_layers=proj_layers,
            use_bn=proj_bn,
        )

    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            h = self.encoder(x.flatten(0, 1))     # (B*T, D_enc)
            h = h.view(B, T, -1).mean(dim=1)      # (B, D_enc)
        else:
            h = self.encoder(x)
        return self.projector(h)


class LinearClassifier(nn.Module):
    """Frozen encoder + linear classifier on mean-pooled frame features."""

    def __init__(self, encoder, num_classes=101):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, clip):
        B, N = clip.shape[0], clip.shape[1]
        x = clip.flatten(0, 1)
        with torch.no_grad():
            feats = self.encoder(x)
        feats = feats.view(B, N, -1).mean(dim=1)
        return self.fc(feats)


if __name__ == '__main__':
    m = SSLModel(proj_dim=128, proj_hidden=512, proj_layers=2, proj_bn=False)
    x = torch.randn(2, 3, 224, 224)
    print('frame mode:', m(x).shape)       # (2, 128)
    x_clip = torch.randn(2, 4, 3, 224, 224)
    print('clip mode :', m(x_clip).shape)  # (2, 128)

    lc = LinearClassifier(m.encoder, num_classes=101)
    print('probe     :', lc(torch.randn(2, 8, 3, 224, 224)).shape)
