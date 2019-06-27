__all__ = ['OctaveConv']
from torch import nn


class OctaveConv(nn.Module):
    def __init__(self, in_channels, channels, alpha_in, alpha_out,
                 kernel_size, padding=0, stride=1, groups=1, bias=False):
        super().__init__()
        assert stride in (1, 2), 'stride should be 1 or 2.'
        assert 0 <= alpha_in < 1 and 0 <= alpha_out < 1, 'Wrong setting with alpha'
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.stride = stride
        self.depth_wise = depth_wise = True if channels == groups else False
        h_in = int((1 - alpha_in) * in_channels)
        l_in = int(alpha_in * in_channels)
        h_out = int((1 - alpha_out) * channels)
        l_out = int(alpha_out * channels)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.W_HH = nn.Conv2d(h_in, h_out, kernel_size, 1, padding, 1, min(h_in, groups), bias)
        if alpha_out != 0 and alpha_in != 0:
            self.W_LL = nn.Conv2d(l_in, l_out, kernel_size, 1, padding, 1, min(l_in, groups), bias)
        if alpha_out != 0 and not depth_wise:
            self.W_HL = nn.Conv2d(h_in, l_out, kernel_size, 1, padding, 1, min(h_in, groups), bias)
        if alpha_in != 0 and not depth_wise:
            self.W_LH = nn.Conv2d(l_in, h_out, kernel_size, 1, padding, 1, min(l_in, groups), bias)

    def forward(self, x_h, x_l=None):
        # vanilla layer
        if self.alpha_in == self.alpha_out == 0:
            y_hh = self.W_HH(x_h) if self.stride == 1 else self.W_HH(self.downsample(x_h))
            return y_hh, None
        # first oct layer(first layer should not be depth wise layer)
        elif self.alpha_in == 0:
            x_h = x_h if self.stride == 1 else self.downsample(x_h)
            y_hh = self.W_HH(x_h)
            y_hl = self.W_HL(self.downsample(x_h))
            return y_hh, y_hl
        # last oct layer
        elif self.alpha_out == 0:
            y_hh = self.W_HH(x_h) if self.stride == 1 else self.W_HH(self.downsample(x_h))
            if not self.depth_wise:
                y_lh = self.upsample(self.W_LH(x_l)) if self.stride == 1 else self.W_LH(x_l)
                y_h_out = y_hh + y_lh
            else:
                y_h_out = y_hh
            return y_h_out, None
        # oct layer
        else:
            y_hh = self.W_HH(x_h) if self.stride == 1 else self.W_HH(self.downsample(x_h))
            y_ll = self.W_LL(x_l) if self.stride == 1 else self.W_LL(self.downsample(x_l))
            if not self.depth_wise:
                y_lh = self.upsample(self.W_LH(x_l)) if self.stride == 1 else self.W_LH(x_l)
                x_h = x_h if self.stride == 1 else self.downsample(x_h)
                y_hl = self.W_HL(self.downsample(x_h))
                y_h_out = y_hh + y_lh
                y_l_out = y_ll + y_hl
            else:
                y_h_out = y_hh
                y_l_out = y_ll
            return y_h_out, y_l_out
