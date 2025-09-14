# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ----------------- Normalization helper -----------------


def Norm3d(ch, use_gn=False):
    return nn.GroupNorm(8, ch) if use_gn else nn.InstanceNorm3d(ch)

# ----------------- Residual convolution block -----------------


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = Norm3d(out_ch, use_gn)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = Norm3d(out_ch, use_gn)
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.Conv3d(
            in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        def _inner_forward(x):
            y = self.relu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return self.relu(y + self.res(x))
        if self.training:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)

# ----------------- Encoder stack -----------------


class Encoder(nn.Module):
    def __init__(self, in_ch, feats=(24, 48, 96), use_gn=False):
        super().__init__()
        layers = []
        for f in feats:
            layers += [ResidualConvBlock(in_ch, f, use_gn), nn.MaxPool3d(2)]
            in_ch = f
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skips = []
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            skips.append(x)
            x = self.layers[i+1](x)
        return skips

# ----------------- CBAM Attention -----------------


class CBAM3D(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.max = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(ch, ch//red, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(ch//red, ch, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
        self.spat = nn.Sequential(
            nn.Conv3d(1, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        x = x * c
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        combined = (avg_map + max_map) / 2
        return x * self.spat(combined)

# ----------------- Modality skip fusion -----------------


class SkipFusionBlock(nn.Module):
    def __init__(self, ch, use_gn=False):
        super().__init__()
        self.g1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, ch//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(ch//16, ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.g2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, ch//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(ch//16, ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.g3 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, ch//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(ch//16, ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv3d(3*ch, ch, 1, bias=False)
        self.norm = Norm3d(ch, use_gn)

    def forward(self, f1, f2, f3):
        f1 = f1 * self.g1(f1)
        f2 = f2 * self.g2(f2)
        f3 = f3 * self.g3(f3)
        return self.norm(self.conv(torch.cat([f1, f2, f3], dim=1)))

# ----------------- Decoder with deep supervision -----------------


class Decoder(nn.Module):
    def __init__(self, feats=(96, 48, 24), out_ch=4, bottleneck_ch=256, ds_at=(0, 1, 2), use_gn=False):
        super().__init__()
        self.up, self.dec, self.aux = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ds_at = set(ds_at)
        in_feats = [bottleneck_ch] + list(feats[:-1])
        for cin, c in zip(in_feats, feats):
            self.up.append(nn.ConvTranspose3d(cin, c, 2, 2))
            self.dec.append(ResidualConvBlock(c*2, c, use_gn))
            self.aux.append(nn.Conv3d(c, out_ch, 1))
        self.final = nn.Conv3d(feats[-1], out_ch, 1)

    def forward(self, x, skips):
        aux_outs = []
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[i]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(
                    x, size=s.shape[2:], mode='trilinear', align_corners=False)
            x = dec(torch.cat([x, s], 1))
            aux_outs.append(self.aux[i](x) if i in self.ds_at else None)
        return self.final(x), aux_outs

# ----------------- Full UNet Model -----------------


class ModalitySelectiveUNet(nn.Module):
    def __init__(self, feats=(24, 48, 96), bottleneck_ch=256, use_gn=False, ds_at=(0, 1, 2)):
        super().__init__()
        self.encoders = nn.ModuleList(
            [Encoder(1, feats, use_gn) for _ in range(3)])
        self.alpha_mlp = nn.Sequential(
            nn.Conv3d(feats[-1], feats[-1]//4, 1, bias=False), nn.ReLU(),
            nn.Conv3d(feats[-1]//4, 3, 1, bias=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(feats[-1], bottleneck_ch, 1, bias=False),
            CBAM3D(bottleneck_ch),
            ResidualConvBlock(bottleneck_ch, bottleneck_ch, use_gn)
        )
        self.skip_fusers = nn.ModuleList(
            [SkipFusionBlock(ch, use_gn) for ch in feats[::-1]])
        self.decoder = Decoder(
            feats[::-1], out_ch=4, bottleneck_ch=bottleneck_ch, ds_at=ds_at, use_gn=use_gn)

    def forward(self, x):
        m1, m2, m3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        skips_all = [enc(m) for enc, m in zip(self.encoders, (m1, m2, m3))]
        deep = torch.stack([s[-1] for s in skips_all], dim=1)
        pooled = F.adaptive_avg_pool3d(
            deep, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        attn = self.alpha_mlp(pooled.transpose(
            1, 2).unsqueeze(-1).unsqueeze(-1))
        attn = attn.mean(dim=1).squeeze(-1).squeeze(-1)
        alpha = torch.softmax(attn, dim=1).view(-1, 3, 1, 1, 1, 1)
        fused = alpha[:, 0]*skips_all[0][-1] + alpha[:, 1] * \
            skips_all[1][-1] + alpha[:, 2]*skips_all[2][-1]
        x = self.bottleneck(fused)
        fused_skips = [fus(*t) for t, fus in zip(zip(*[reversed(s)
                                                       for s in skips_all]), self.skip_fusers)]
        return self.decoder(x, fused_skips)
