import os
import framework.utils.tensorboard_helpers as mira_th
import matplotlib.pyplot as plt
import numpy as np
import torch

separator = '----------------------------------------'


def mk_dirs(*dirs):
    for i, dir in enumerate(dirs):
        if not os.path.exists(dir):
            os.makedirs(dir)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def norm_ten(pred_ce, binary=False):

    mmax = pred_ce.max()
    mmin = pred_ce.min()
    pred_ce = (pred_ce - mmin)/ (mmax - mmin+ 0.001)
    if binary:
        pred_ce[pred_ce>=0.5] == 1
        pred_ce[pred_ce<0.5] == 0
    return pred_ce

def plot_3d_tensor(x, color=None):
    x = x.cpu().detach().numpy()
    x= (norm_ten(x[0,...]) *255).astype(np.uint8)
    if color is not None:
        plt.imshow(x, cmap=color)
    else:
        plt.imshow(x)
    plt.colorbar()
    plt.show()

def plot_3d_np(x):
    x= (norm_ten(x[0,...]) *255).astype(np.uint8)
    plt.imshow(x)
    plt.colorbar()
    plt.show()

def plot_multi_3d(x_list, row, col, title_list=None, save_path=None):
    assert row*col >= len(x_list)
    fig = plt.figure(figsize=(32, 24))    #
    for i, x in enumerate(x_list):
        x = x.cpu().detach().numpy()
        x = (norm_ten(x[0, ...]) * 255).astype(np.uint8)

        ax = plt.subplot(row, col, i+1)
        if title_list is not None:
            title = title_list[i]
            ax.title.set_text(title)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        ax.imshow(x)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_multi_2d(x_list, row, col, title_list=None, save_path=None):
    assert row*col >= len(x_list)
    fig = plt.figure(figsize=(16, 10))    #
    for i, x in enumerate(x_list):
        ax = plt.subplot(row, col, i+1)
        if title_list is not None:
            title = title_list[i]
            ax.title.set_text(title)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        ax.imshow(x)

    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)

def write_values(writer, phase, value_dict, n_iter):
    for name, value in value_dict.items():
        writer.add_scalar('{}/{}'.format(phase, name), value, n_iter)

def write_images(writer, phase, image_dict, n_iter, mode3d, batch_id=0):
    for name, image in image_dict.items():
        if mode3d:
            if len(image.size()) == 5:
                if image.size(1) == 1:
                    (image), n_iter
                    writer.add_image('{}/{}'.format(phase, name +''), mira_th.normalize_to_0_1(image[batch_id, :, int(image.size(2)/2), ...]), n_iter)
                elif image.size(1) > 3:
                    writer.add_image('{}/{}'.format(phase, name),
                                     torch.clamp(image[batch_id, 1:4, int(image.size(2) / 2), ...], 0, 1), n_iter,
                                     dataformats='CHW')
                elif image.size(1) == 3:
                    writer.add_image('{}/{}'.format(phase, name),
                                     mira_th.normalize_to_0_1(image[batch_id, :, int(image.size(2) / 2), ...]), n_iter,
                                     dataformats='CHW')
                else:
                    writer.add_image('{}/{}'.format(phase, name),
                                     mira_th.normalize_to_0_1(image[batch_id, 1, int(image.size(2) / 2), ...]), n_iter,
                                     dataformats='HW')
            elif len(image.size()) ==4:
                '''typically for showing the displacement field with [batch, 3, x, y]'''
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[batch_id, ...]), n_iter, dataformats='CHW')
        else:
            if image.size(1) ==  1:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[batch_id, ...]), n_iter)
            elif image.size(1) > 3:
                writer.add_image('{}/{}'.format(phase, name), torch.clamp(image[batch_id, 1:4, ...], 0, 1), n_iter, dataformats='CHW')
            elif image.size(1) == 3:
                writer.add_image('{}/{}'.format(phase, name), torch.clamp(image[batch_id, ...], 0, 1), n_iter, dataformats='CHW')
            else:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[batch_id, 1, ...]), n_iter, dataformats='HW')

