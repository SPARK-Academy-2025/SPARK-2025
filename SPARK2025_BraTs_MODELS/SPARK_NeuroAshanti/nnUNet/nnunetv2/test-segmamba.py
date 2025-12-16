# nnUNet/nnunetv2/test-segmamba.py

import torch 
from nnunetv2.model_segmamba.segmamba import SegMamba

t1 = torch.rand(1, 4, 128, 128, 128).cuda()

model = SegMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).cuda()

out = model(t1)

for n in range(len(out)):
    print(out[n].shape)

print(model)