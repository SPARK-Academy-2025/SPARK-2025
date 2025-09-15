"""Contains the architecture for the baseline 3D U-Net model, based on NVIDIA's optimized U-Net."""

import torch
import torch.nn as nn

# instance norm    
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out1,ch_out2,k1,k2,s1,s2):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out1, kernel_size=k1,stride=s1,padding=1,bias=True),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(ch_out1),
            nn.Conv3d(ch_out1, ch_out2, kernel_size=k2,stride=s2,padding=1,bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
      
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            #nn.ConvTranspose3d(ch_in, ch_out,kernel_size=2,stride=2,padding=1,bias=True)
            nn.ConvTranspose3d(ch_in, ch_out,kernel_size=2,stride=2,padding=1,bias=True,output_padding=1,dilation=2),
            #nn.Upsample(scale_factor=2),
            nn.InstanceNorm3d(ch_in),
 	        nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.device)
        x = self.up(x)
        
        return x

# 3d unet with optimized instance norm    
class U_Net3d(nn.Module):
    def __init__(self,img_ch=4,output_ch=3):
        super(U_Net3d,self).__init__()
        nf= 32 #options are: 8, 16, 64
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2).to(device='cuda:0')
        self.Conv1 = conv_block(ch_in=img_ch,ch_out1=nf*2,ch_out2=nf*2,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        self.Conv2 = conv_block(ch_in=nf*2,ch_out1=nf*3,ch_out2=nf*3,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')
        self.Conv3 = conv_block(ch_in=nf*3,ch_out1=nf*4,ch_out2=nf*4,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')
        self.Conv4 = conv_block(ch_in=nf*4,ch_out1=nf*6,ch_out2=nf*6,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')
        self.Conv5 = conv_block(ch_in=nf*6,ch_out1=nf*8,ch_out2=nf*8,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')
        self.Conv6 = conv_block(ch_in=nf*8,ch_out1=nf*12,ch_out2=nf*12,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')
        self.Conv7 = conv_block(ch_in=nf*12,ch_out1=nf*16,ch_out2=nf*16,k1=3,k2=3,s1=2,s2=1).to(device='cuda:0')

        self.Up6 = up_conv(ch_in=nf*16,ch_out=nf*12).to(device='cuda:0')
        self.Up_conv6 = conv_block(ch_in=nf*24, ch_out1=nf*12, ch_out2=nf*12,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        
        self.Up5 = up_conv(ch_in=nf*12,ch_out=nf*8).to(device='cuda:0')
        self.Up_conv5 = conv_block(ch_in=nf*16, ch_out1=nf*8, ch_out2=nf*8,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')

        self.Up4 = up_conv(ch_in=nf*8,ch_out=nf*6).to(device='cuda:0')
        self.Up_conv4 = conv_block(ch_in=nf*12, ch_out1=nf*6, ch_out2=nf*6,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        
        self.Up3 = up_conv(ch_in=nf*6,ch_out=nf*4).to(device='cuda:0')
        self.Up_conv3 = conv_block(ch_in=nf*8, ch_out1=nf*4,ch_out2=nf*4,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        self.Conv_1x13 = nn.Conv3d(nf*4,output_ch,kernel_size=1,stride=1,padding=0).to(device='cuda:0')
        
        self.Up2 = up_conv(ch_in=output_ch,ch_out=nf*3).to(device='cuda:0')
        self.Up_conv2 = conv_block(ch_in=nf*6 , ch_out1=nf*3,ch_out2=nf*3,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        self.Conv_1x12 = nn.Conv3d(nf*3,output_ch,kernel_size=1,stride=1,padding=0).to(device='cuda:0')
        
        self.Up1 = up_conv(ch_in=output_ch,ch_out=nf*2).to(device='cuda:0')
        self.Up_conv1 = conv_block(ch_in=nf*4, ch_out1=nf*2,ch_out2=nf*2,k1=3,k2=3,s1=1,s2=1).to(device='cuda:0')
        self.Conv_1x11 = nn.Conv3d(nf*2,output_ch,kernel_size=1,stride=1,padding=0).to(device='cuda:0')
        
        self.Sig =nn.Sigmoid().to(device='cuda:0')


    def forward(self,x):
        # encoding path
        x = x.to(device='cuda:0')
        x1 = self.Conv1(x)
 
        x2 = self.Conv2(x1)       
        
        x3 = self.Conv3(x2)
       
        x4 = self.Conv4(x3)       
       
        x5 = self.Conv5(x4)       
        
        x6 = self.Conv6(x5)

        x7 = self.Conv7(x6)
        

        # decoding + concat path
       
        d6 = self.Up6(x7.to(device='cuda:0'))
        d6 = torch.cat((x6.to(device='cuda:0'),d6),dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x5.to(device='cuda:0'),d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x4.to(device='cuda:0'),d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3.to(device='cuda:0'),d3),dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.Conv_1x13(d3)
        d3 = self.Sig(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x2.to(device='cuda:0'),d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.Conv_1x12(d2)
        d2 = self.Sig(d2)
        
        d1 = self.Up1(d2)
        d1 = torch.cat((x1.to(device='cuda:0'),d1),dim=1)
        d1 = self.Up_conv1(d1)
        d1 = self.Conv_1x11(d1)
        d1 = self.Sig(d1)

        return d1.to(device='cuda:0')
    
    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return f"UNet_3d (with {num_params:,} parameters)"