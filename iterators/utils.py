import os
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

from edflow.util import walk

def pt2np(tensor, permute=True):
    '''Converts a torch Tensor to a numpy array.'''
    array = tensor.detach().cpu().numpy()
    if permute:
        array = np.transpose(array, (0, 2, 3, 1))
    return array

def set_gpu(config):
    """Move the model to device cuda if available and use the specified GPU"""
    if "CUDA_VISIBLE_DEVICES" in config:
        if type(config["CUDA_VISIBLE_DEVICES"]) != str:
            config["CUDA_VISIBLE_DEVICES"] = str(config["CUDA_VISIBLE_DEVICES"])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_state(random_seed):
    '''Set random seed for torch and numpy.'''
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def calculate_gradient_penalty(discriminator, real_images, fake_images, device):
    '''Return the gradient penalty for the discriminator.'''
    eta = torch.FloatTensor(real_images.size()[0], 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images.size()[0], real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)
    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def convert_logs2numpy(logs):
    def conditional_convert2np(log_item):
        if isinstance(log_item, torch.Tensor):
            log_item = log_item.detach().cpu().numpy()
        return log_item
    # convert to numpy
    walk(logs, conditional_convert2np, inplace=True)
    return logs