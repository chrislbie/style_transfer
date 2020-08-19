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

def conditional_intance_normalization(cont, mean, std):
    """Renormalize content with certain mean and standard deviation

    Args:
        cont (torch.Tensor): content of 2 images, shape (N=2, C, H, W)
        mean (torch.Tensor): mean of style vector of 2 images, shape (N=2, 1, 1, 1)
        std (torch.Tensor): standard deviation of style vector of 2 images, shape (N=2, 1, 1, 1)

    Returns:
        torch.Tensor: normalized content in every combination
    """
    cs = []
    for m, s in zip(mean, std):
        for c in cont:
            cs.append(c * m / s)
    cs = torch.stack(cs)
    return cs