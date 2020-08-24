import torch
import torch.nn as nn

class Transpose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=True, kernel_size=4, stride=2, padding=1):
        super(Transpose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d( in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)


class ExtraConvBlock(nn.Module):
    def __init__(self, channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=False):
        super(ExtraConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=True, kernel_size=3, stride=2, padding=1):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        layers.append(Activation)

        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation, batch_norm=False, drop_rate=None, bias=True):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_channels, out_channels, bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        if Activation is not None:
            layers.append(Activation)   
        self.main = nn.Sequential(*layers) 
    def forward(self, x):
        return self.main(x)

class ConditionalInstanceNormalization(nn.Module):
    """Renormalize content to scale with offest defined by style vector

    Args:
        content_channels (int): Number of channels of content input
        style_dim (int): Dimension of style vectors
    """
    def __init__(self, content_channels, style_dim):
        super(ConditionalInstanceNormalization, self).__init__()
        self.scale = LinearBlock(style_dim, content_channels, nn.LeakyReLU(0.2))
        self.offset = LinearBlock(style_dim, content_channels, nn.LeakyReLU(0.2))
        self.inst = nn.InstanceNorm2d(content_channels, momentum=0)
    
    def forward(self, content, style):
        scale = self.scale(style)[..., None, None]
        offset = self.offset(style)[..., None, None]
        content = self.inst(content)
        content = content * scale + offset
        
        return content

class StyleResidualBlock(nn.Module):
    def __init__(self, channels, style_dim,  Activation=nn.LeakyReLU(0.2)):
        super(StyleResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.cond_inst_norm1 = ConditionalInstanceNormalization(channels, style_dim)
        self.cond_inst_norm2 = ConditionalInstanceNormalization(channels, style_dim)
        
        self.act = Activation

    def forward(self, x, style):
        y = self.act(self.cond_inst_norm1(self.conv1(x), style))
        y = self.act(self.cond_inst_norm2(self.conv2(y), style))

        return x + y
