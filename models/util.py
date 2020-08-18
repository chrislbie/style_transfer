import torch

def style_mean_std(s, epsilon=1e-5):

    """[Calculate means of the style vectors]

    Args:
        s ([torch.Tensor]): [Style vector of shape (N, D)]
        epsilon([float]): [Offset for standard deviance]
    
    Retruns:
        [torch.Tensor]: [mean of style vectors, shape (N, 1, 1, 1)]
        [torch.Tensor]: [std of style vectors, shape (N, 1, 1, 1)}
    """
    print(s.shape)
    mean = torch.mean(s, dim=1)
    std = torch.std(s, dim=1)
    
    return mean, std
