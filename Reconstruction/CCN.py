import torch
from torch import nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.conv_imag = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)

    def forward(self, input_real, input_imag):
        assert input_real.shape == input_imag.shape
        return (self.conv_real(input_real) - self.conv_imag(input_imag)), (
                    self.conv_imag(input_real) + self.conv_real(input_imag))


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_tran_real = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_imag = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input_real, input_imag):
        assert input_real.shape == input_imag.shape
        return (self.conv_tran_real(input_real) - self.conv_tran_imag(input_imag)), (self.conv_tran_imag(input_real) + self.conv_tran_real(input_imag))


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output


def complex_max_pool2d(input_real, input_imag, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    # input=input_real + 1j*input_imag
    real_value, indices = F.max_pool2d(
        input_real,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True
    )
    real_value = real_value
    ang = torch.atan2(input_imag, input_real)
    ang = _retrieve_elements_from_indices(ang, indices)
    imag_value = (real_value * torch.tan(ang))
    return real_value, imag_value


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input_real, input_imag):
        return complex_max_pool2d(input_real, input_imag, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)


class ComplexPReLU(nn.Module):
    def __init__(self, slope=0.2):
        super(ComplexPReLU, self).__init__()
        self.slope = slope

    def lrelu(self, x):
        outt = torch.max(self.slope * x, x)
        return outt

    def forward(self, x_real, x_imag):
        return self.lrelu(x_real), self.lrelu(x_imag)


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_real, x_imag):
        return self.relu(x_real), self.relu(x_imag)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x_real, x_imag):
        x_real = self.bn_re(x_real)
        x_imag = self.bn_im(x_imag)
        return x_real, x_imag

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNorm2d(out_ch)
        self.relu = ComplexReLU()

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv(x_real, x_imag)
        x_real, x_imag = self.relu(x_real, x_imag)
        x_real, x_imag = self.bn(x_real, x_imag)
        return x_real, x_imag


class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = ComplexConv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = ComplexBatchNorm2d(ch)
        self.relu = ComplexReLU()
        self.conv2 = ComplexConv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = ComplexBatchNorm2d(ch)

    def forward(self, x_real, x_imag):
        res_real, res_imag = x_real, x_imag
        res_real, res_imag = self.bn1(res_real, res_imag)
        res_real, res_imag = self.relu(res_real, res_imag)
        res_real, res_imag = self.conv1(res_real, res_imag)
        # res_real, res_imag = self.bn2(res_real, res_imag)
        # res_real, res_imag = self.conv2(res_real, res_imag)
        return res_real+x_real, res_imag+x_imag


class resnet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, depth=5, ngf=256, Nz_im=None, Nx_im=None, sample=False, mask=None, freeze_mask=False, Nz=None, Nx=None):
        super(resnet, self).__init__()
        self.Nz_im = Nz_im
        self.Nx_im = Nx_im
        self.Nz = Nz
        self.Nx = Nx
        self.sample = sample
        if sample:
            self.mask = nn.Parameter(torch.rand(self.Nz, self.Nx))
            self.mask.data = torch.from_numpy(mask.astype('float32'))
            if freeze_mask:
                self.mask.requires_grad = False

        self.depth = depth
        self.head = ConvBNReLU(in_ch=in_channel, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.body1 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        
        if self.depth >= 2:
            self.body2 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 3:
            self.body3 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 4:
            self.body4 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 5:
            self.body5 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 6:
            self.body6 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 7:
            self.body7 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 8:
            self.body8 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 9:
            self.body9 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        if self.depth >= 10:
            self.body10 = ResBlock(ngf, kernel_size=3, stride=1, padding=1)

        self.tail = ComplexConv2d(ngf, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x_real, x_imag):
        if self.sample:
            x_real = x_real * self.mask
            x_imag = x_imag * self.mask

        x_real, x_imag = self.head(x_real, x_imag)
        x_real, x_imag = self.body1(x_real, x_imag)

        if self.depth >= 2:
            x_real, x_imag = self.body2(x_real, x_imag)
        if self.depth >= 3:
            x_real, x_imag = self.body3(x_real, x_imag)
        if self.depth >= 4:
            x_real, x_imag = self.body4(x_real, x_imag)
        if self.depth >= 5:
            x_real, x_imag = self.body5(x_real, x_imag)
        if self.depth >= 6:
            x_real, x_imag = self.body6(x_real, x_imag)
        if self.depth >= 7:
            x_real, x_imag = self.body7(x_real, x_imag)
        if self.depth >= 8:
            x_real, x_imag = self.body8(x_real, x_imag)
        if self.depth >= 9:
            x_real, x_imag = self.body9(x_real, x_imag)
        if self.depth >= 10:
            x_real, x_imag = self.body10(x_real, x_imag)

        if (self.Nz is not None) & (self.Nx is not None):
            x_real = F.interpolate(x_real, size=(self.Nz_im, self.Nx_im), mode='bicubic')
            x_imag = F.interpolate(x_imag, size=(self.Nz_im, self.Nx_im), mode='bicubic')

        x_real, x_imag = self.tail(x_real, x_imag)
        
        if self.sample:
            return x_real, x_imag, self.mask
        else:
            return x_real, x_imag


if __name__ == '__main__':
    import torch
    net = resnet(depth=1)
    x_real = torch.rand(1, 1, 128, 128)
    x_imag = torch.rand(1, 1, 128, 128)
    y_real, y_imag = net(x_real, x_imag)
    print(net)
