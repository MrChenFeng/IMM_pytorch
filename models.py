import torch.nn as nn
import torch 

def get_gaussian_mean(x, axis, other_axis):
    u = torch.softmax(torch.sum(x, axis=other_axis), axis=2)
    ind = torch.arange(x.shape[axis]).cuda()
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position


class Regressor(nn.Module):
    def __init__(self, dim, ind):
        super(Regressor, self).__init__()
        self.linear = nn.Linear(512, 2 * dim)
        self.ind = float(ind)
        self.dim = dim

    def forward(self, x):
        h_axis = 2
        w_axis = 3
        h_mean = get_gaussian_mean(x, h_axis, w_axis) / self.ind  # .unsqueeze(-1)
        w_mean = get_gaussian_mean(x, w_axis, h_axis) / self.ind  # .unsqueeze(-1)

        gaussian = torch.cat([h_mean, w_mean], axis=1)  # Bx512

        res = torch.sigmoid(self.linear(gaussian))

        return res.view(-1, self.dim, 2)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape,self).__init__()
        self.shape = shape
    def __call__(self, x):
        return x.view(self.shape)
    
class LinearRegressor(nn.Module):
    def __init__(self, dim):
        super(LinearRegressor, self).__init__()
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 2 * dim)
        #self.reshape = Reshape((batchsize, dim, 2))
        self.dim = dim
    def forward(self, x):
        res = torch.relu(self.linear1(x))
        res = torch.relu(self.linear2(res))
        return res.view(-1, self.dim, 2)