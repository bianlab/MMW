import torch
from torch.nn import functional as F

def getImsize(x):
    b, c = x.size()[0], x.size()[1]
    h, w = x.size()[2], x.size()[3]
    return b, c, h, w


def getTVmap(x, type):
    b, c, h, w = getImsize(x)
    device = x.device
    if type == 1:
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w - 1])
    elif type == 2:
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2)
    elif type == 3:
        dh = torch.Tensor([[[[1, 2, 2, 1], [-1, -2, -2, -1]]]])/12.0
        dw = torch.Tensor([[[[1, -1], [2, -2], [2, -2], [1, -1]]]]) / 12.0
        dh, dw = dh.repeat(1, c, 1, 1), dw.repeat(1, c, 1, 1)
        dh, dw = dh.to(device), dw.to(device)
        h_tv = torch.abs(F.conv2d(x, dh))
        w_tv = torch.abs(F.conv2d(x, dw))
    elif type == 4:
        dh = torch.Tensor([[[[-1], [2], [-1]]]])
        dw = torch.Tensor([[[[-1, 2, -1]]]])
        dh, dw = dh.repeat(1, c, 1, 1), dw.repeat(1, c, 1, 1)
        dh, dw = dh.to(device), dw.to(device)
        h_tv = torch.pow(F.conv2d(x, dh, padding=(1, 0)), 2)
        w_tv = torch.pow(F.conv2d(x, dw, padding=(0, 1)), 2)
    else:
        h_tv, w_tv = 0, 0
    return h_tv, w_tv


class TVloss(torch.nn.Module):
    def __init__(self, type=1):
        super(TVloss, self).__init__()
        self.type = type

    def forward(self, x):
        b, c, h, w = getImsize(x)
        count_h = (h - 1) * w
        count_w = h * (w - 1)
        h_tv, w_tv = getTVmap(x, self.type)
        h_tv, w_tv = h_tv.sum(), w_tv.sum()
        loss = 2*(h_tv/count_h + w_tv/count_w)/b
        return loss