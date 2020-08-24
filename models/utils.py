import torch

def style_mean_std(s, epsilon=1e-5):

    """Calculate means of the style vectors

    Args:
        s (torch.Tensor): Style vector of shape (N, D)
        epsilon(float): Offset for standard deviantion
  
    Retruns:
        torch.Tensor: mean of style vectors, shape (N, 1, 1, 1)
        torch.Tensor: std of style vectors, shape (N, 1, 1, 1)
    """
    mean = torch.mean(s, dim=1)[:, None, None, None]
    std = torch.std(s, dim=1)[:, None, None, None] + epsilon
    
    return mean, std

def l2_normalize(x):
    x = x/torch.sqrt(torch.sum(x**2, dim=-1))[...,None]
    return x