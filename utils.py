import torch

def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp[:, :, 0], height)
    wm = squared_diff(kp[:, :, 1], width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + eps).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm

def get_gaussian_mean(x, axis, other_axis):
    """

    Args:
        x (float): Input images(BxCxHxW)
        axis (int): The index for weighted mean
        other_axis (int): The other index

    Returns: weighted index for axis, BxC

    """
    u = torch.softmax(torch.sum(x, axis=other_axis), axis=2)
    size = x.shape[axis]
    ind = torch.linspace(-1,1,size).to(x.device)
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position