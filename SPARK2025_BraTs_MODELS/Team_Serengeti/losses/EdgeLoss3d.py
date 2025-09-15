# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 2022
Last Modified on Thu Apr 6 2023
@author: Agamdeep Chopra, achopra4@uw.edu
@affiliation: University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection
            from scratch with PyTorch. Medium. Retrieved July 10, 2022, from
            https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import general_losses as lf

EPSILON = 1E-6


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, and diagonal kernels
    ex:
        Sx = [[[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]],
              [[-n2, 0, n2],
               [-n2*n3, 0, n2*n3],
               [-n2, 0, n2]],
              [[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]]]
    Parameters
    ----------
    n1 : int, optional
        kernel value 1. The default is 1.
    n2 : int, optional
        kernel value 2. The default is 2.
    n3 : int, optional
        kernel value 3. The default is 2.
    Returns
    -------
    list
        list of all the 3d sobel kernels.
    '''
    Sx = np.asarray([[[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]], [[-n2, 0, n2],
                    [-n3*n2, 0, n3*n2], [-n2, 0, n2]], [[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]]])
    Sy = np.asarray([[[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]], [[-n2, -n3*n2, -n2],
                    [0, 0, 0], [n2, n3*n2, n2]], [[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]]])
    Sz = np.asarray([[[-n1, -n2, -n1], [-n2, -n3*n2, -n2], [-n1, -n2, -n1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[n1, n2, n1], [n2, n3*n2, n2], [n1, n2, n1]]])
    Sd11 = np.asarray([[[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]], [[0, n2, n2*n3],
                      [-n2, 0, n2], [-n2*n3, -n2, 0]], [[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]]])
    Sd12 = np.asarray([[[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]], [[-n2*n3, -n2, 0],
                      [-n2, 0, n2], [0, n2, n2*n3]], [[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]]])
    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = np.asarray([-S.T for S in Sd11.T])
    Sd32 = np.asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    '''
    Sobel edge detection algorithm compatible with PyTorch Autograd engine.
    '''

    def __init__(self, n1=1, n2=2, n3=2, device='cuda:1'):
        super(GradEdge3D, self).__init__()
        self.device = device
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        for s in S:
            sobel_filter = nn.Conv3d(in_channels=1, out_channels=1, stride=1,
                                     kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = torch.from_numpy(
                s.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(device, dtype=torch.float32)
            self.sobel_filters.append(sobel_filter)

    def detect(self, img, a=1):
        '''
        Detect edges using Sobel operator for a 3d image
        Parameters
        ----------
        img : torch tensor
            3D torch tensor of shape (b, c, x, y, z).
        a : int, optional
            padding to be added, do not change unless necessary. The default is 1.
        Returns
        -------
        torch tensor
            tensor of gradient edges of shape (b, 1, x, y, z).
        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = nn.functional.pad(img, pad, mode='reflect')

        grad_mag = (1 / C) * torch.sum(torch.stack([torch.sum(torch.cat([s(img[:, c:c+1])for c in range(
            C)], dim=1) + EPSILON, dim=1) ** 2 for s in self.sobel_filters], dim=1) + EPSILON, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        return grad_mag.view(B, 1, H, W, D)


class GMELoss3D(nn.Module):
    '''
    3D-Edge Loss for PyTorch with choice of criterion. Default is MSELoss.
    '''

    def __init__(self, criterion=lf.NCCLoss(), n1=1, n2=2, n3=2, device='cuda:1'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3, device)
        self.criterion = criterion

    def forward(self, y, yp):
        y_edge = self.edge_filter.detect(y)
        yp_edge = self.edge_filter.detect(yp)
        error = self.criterion(y_edge, yp_edge)
        return error


if __name__ == "__main__":
    device = 'cuda'
    loss = GMELoss3D(device=device)
    filter_ = GradEdge3D(n1=1, n2=2, n3=2, device=device)
    plt.rcParams['figure.dpi'] = 150

    for k in range(1, 5):
        path = 'R:/img (%d).pkl' % (k)
        # T1 and T2 combined(ie- 2 channel input). For single channel add [0:1] to the line after calling np.load
        data = np.load(path, allow_pickle=True)
        x = torch.from_numpy(data[0]).view(
            1, 1, data[0].shape[0], data[0].shape[1], data[0].shape[2]).to(device=device, dtype=torch.float)
        y = filter_.detect(x)
        x, y = (x - torch.min(x))/(torch.max(x) - torch.min(x)
                                   ), (y - torch.min(y))/(torch.max(y) - torch.min(y))
        Y = [x, y]
        y = torch.cat([0.3 * Y[0], 0.3 * Y[0] + 0.7 * Y[1],
                      0.3 * Y[0]], dim=1).squeeze().cpu().detach()
        print(y.shape)
        titles = ['input + grad_magnitude']

        for j in range(0, 150, 1):
            out = y[:, :, :, j]
            plt.imshow(out.numpy().squeeze().T)
            plt.title(titles[0] + ' slice %d' % (j))
            plt.show()

    print('test_loss =', loss(x, x + 0.001 *
          torch.rand(x.shape).to(device=device, dtype=torch.float)))