import matplotlib.pyplot as plt
import torch.nn as nn


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for weight in m.parameters():
        if l2_reg is None:
            l2_reg = weight.norm(2)
        else:
            l2_reg = l2_reg + weight.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iteration):
    plt.imshow(pred[0, :, :], cmap='Greys')
    plt.savefig(f'images/{iteration}_prediction.png')
    plt.imshow(mask[0, :, :], cmap='Greys')
    plt.savefig(f'images/{iteration}_mask.png')
