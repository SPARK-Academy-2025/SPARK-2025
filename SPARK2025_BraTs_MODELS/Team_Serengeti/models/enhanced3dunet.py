import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=False)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm3d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return F.leaky_relu(out, inplace=False)

class Optimized3DUNet(nn.Module):
    def __init__(self, img_ch=4, output_ch=3):
        super().__init__()
        nf = 8  # Increased base filters
        self.deep_supervision = True
        
        # Encoder with residual blocks
        self.Conv1 = self._make_residual_block(img_ch, nf)
        self.Conv2 = self._make_residual_block(nf, nf*2, stride=2)
        self.Conv3 = self._make_residual_block(nf*2, nf*4, stride=2)
        self.Conv4 = self._make_residual_block(nf*4, nf*8, stride=2)
        self.Conv5 = self._make_residual_block(nf*8, nf*16, stride=2)
        
        # Decoder with attention gates
        self.Up4 = self._make_up_block(nf*16, nf*8)
        self.Att4 = AttentionGate3D(F_g=nf*8, F_l=nf*8, F_int=nf*4)
        self.Up_conv4 = self._make_residual_block(nf*16, nf*8)
        
        self.Up3 = self._make_up_block(nf*8, nf*4)
        self.Att3 = AttentionGate3D(F_g=nf*4, F_l=nf*4, F_int=nf*2)
        self.Up_conv3 = self._make_residual_block(nf*8, nf*4)
        
        self.Up2 = self._make_up_block(nf*4, nf*2)
        self.Att2 = AttentionGate3D(F_g=nf*2, F_l=nf*2, F_int=nf)
        self.Up_conv2 = self._make_residual_block(nf*4, nf*2)
        
        self.Up1 = self._make_up_block(nf*2, nf)
        self.Att1 = AttentionGate3D(F_g=nf, F_l=nf, F_int=nf//2)
        self.Up_conv1 = self._make_residual_block(nf*2, nf)
        
        # Output layers with sigmoid activation
        self.out_conv1 = nn.Sequential(
            nn.Conv3d(nf*8, output_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv3d(nf*4, output_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv3 = nn.Sequential(
            nn.Conv3d(nf*2, output_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv3d(nf, output_ch, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout3d(0.2)
        
    def _make_residual_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            ResidualConvBlock(in_ch, out_ch, stride),
            nn.Dropout3d(0.2)
        )
        
    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=False),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x, deep_supervision=None):
        deep_supervision = self.deep_supervision if deep_supervision is None else deep_supervision
        
        # Encoder
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        
        # Decoder with attention
        d4 = self.Up4(x5)
        x4_att = self.Att4(g=d4, x=x4)
        d4 = torch.cat([x4_att, d4], dim=1)
        d4 = self.Up_conv4(d4)
        out4 = self.out_conv1(d4)
        
        d3 = self.Up3(d4)
        x3_att = self.Att3(g=d3, x=x3)
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.Up_conv3(d3)
        out3 = self.out_conv2(d3)
        
        d2 = self.Up2(d3)
        x2_att = self.Att2(g=d2, x=x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.Up_conv2(d2)
        out2 = self.out_conv3(d2)
        
        d1 = self.Up1(d2)
        x1_att = self.Att1(g=d1, x=x1)
        d1 = torch.cat([x1_att, d1], dim=1)
        d1 = self.Up_conv1(d1)
        out1 = self.Conv_1x1(d1)
        
        if deep_supervision:
            return {
                'final': self.dropout(out1),
                'ds3': F.interpolate(self.dropout(out2), scale_factor=2, mode='trilinear'),
                'ds2': F.interpolate(self.dropout(out3), scale_factor=4, mode='trilinear'),
                'ds1': F.interpolate(self.dropout(out4), scale_factor=8, mode='trilinear')
            }
        return self.dropout(out1)