import torch.nn as nn
import torch

class UNet3D(nn.Module):
    """
    3D UNet Model
    """
    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128]):
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool3d(2)(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = self._pad(x, skip)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def _pad(self, x, ref):
        diffZ = ref.size(2) - x.size(2)
        diffY = ref.size(3) - x.size(3)
        diffX = ref.size(4) - x.size(4)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2,
                                  diffZ // 2, diffZ - diffZ // 2])
        return x

class IdentityUNet(nn.Module):
    """
    IdentityUNet to test the pipeline
    """
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x[:, 4:, :, :, :]