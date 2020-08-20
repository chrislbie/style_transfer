import torch
import torch.nn as nn

class Transpose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=False, kernel_size=4, stride=2, padding=1):
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
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=False, kernel_size=3, stride=2, padding=1):
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

class Conv2dBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, Activation=nn.ReLU(), batch_norm=False, drop_rate=None, bias=False, kernel_size=3, stride=2, padding=1, pooling=None, num_extra_conv=0, residual=False):
        super(Conv2dBlock2, self).__init__()
        assert (residual and pooling is not None) or not residual, "residual only works with pooling"
        stride = stride if pooling is None else 1
        self.residual = residual
        self.pooling = pooling
        #Blocks to preserve spatial size
        for i in range(num_extra_conv):
            layers = [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=bias)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(in_channels))
            if drop_rate is not None:
                layers.append(nn.Dropout(drop_rate))
            layers.append(Activation)

        #Blocks to reduce spatial size
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if drop_rate is not None:
            layers.append(nn.Dropout(drop_rate))
        
        self.main = nn.Sequential(*layers)
        self.act = Activation
        
        if pooling is not None:
            if pooling == 'max':
                self.pooling_layer = nn.MaxPool2d(2)
            elif pooling == 'avg':
                self.pooling_layer = nn.AvgPool2d(2)
            else:
                assert False, "No valid pooling argument. Valid options: None (default), 'max', 'avg'"
        
        if self.residual:
            self.downsample = nn.AvgPool2d(2)
            res_layers = [nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias)]
            if batch_norm:
                res_layers.append(nn.BatchNorm2d(out_channels))
            if drop_rate is not None:
                res_layers.append(nn.Dropout(drop_rate))
            self.res_layers = nn.Sequential(*res_layers)
        
    def forward(self, x):
        y = self.main(x)

        if self.residual:
            y = self.act(y)
            y = torch.cat((y, x), dim=1)
            y = self.res_layers(y)
        
        if self.pooling:
            y = self.pooling_layer(y)

        return self.act(y)        

inp = torch.ones((5, 8, 64, 64))
block = Conv2dBlock2(8, 16, residual=True, pooling='max')
print(block)
print(block(inp).shape)
