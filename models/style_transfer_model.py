import torch
import torch.nn as nn
import numpy as np
import sys
import os
import yaml
from edflow import get_logger


try:
    from models.modules import Transpose2dBlock, Conv2dBlock
    from models.utils import style_mean_std, conditional_intance_normalization
except ModuleNotFoundError:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cur_path = cur_path.replace("\\", "/")
    path = cur_path[:cur_path.rfind('/')]
    sys.path.append(path)
    from models.modules import Transpose2dBlock, Conv2dBlock
    from models.utils import style_mean_std, conditional_intance_normalization

class Style_Transfer_Model(nn.Module):
    def __init__(self,
                min_channels,
                max_channels,
                in_channels,
                out_channels,
                in_size,
                num_classes,
                block_activation=nn.ReLU(),
                final_activation=nn.Tanh(),
                batch_norm=False,
                drop_rate=None,
                bias=True):
        
        """This is the constructor for the full style transfer model with costum style and content encoder and decoder.

        Args:
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'.
            in_channels (int): Channel dimension of the input image.
            out_channels (int): Channel dimension of the output image.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            final_activation (torch.nn module, optional): Activation function used in the last convolution for the output image. Defaults to nn.Tanh().            
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(Style_Transfer_Model, self).__init__()
        self.logger = get_logger("Style_Transfer_Model")
        
        self.c_enc = Content_Encoder(min_channels, max_channels, in_channels, block_activation, batch_norm, drop_rate, bias)
        self.s_enc = Style_Encoder(min_channels, max_channels, in_channels, block_activation, batch_norm, drop_rate, bias)
        self.dec = Decoder(min_channels, max_channels, out_channels, block_activation, final_activation, batch_norm, drop_rate, bias)
        self.inst_norm = nn.InstanceNorm2d(max_channels, momentum=0.)

        self.disc = Discriminator(out_channels, in_size, min_channels, max_channels, num_classes, nn.LeakyReLU(), batch_norm, drop_rate, bias)
        
        self.logger.info("initialized.")

    def forward(self, x, y):
        assert (x.shape[0] == 2 & y.shape[0] == 2)
        self.c = self.c_enc(x)
        self.s = self.s_enc(y)
        mean, std = style_mean_std(self.s)
        self.c_norm = self.inst_norm(self.c)
        self.cs = conditional_intance_normalization(self.c_norm, mean, std)
        return self.dec(self.cs)

class Content_Encoder(nn.Module):
    def __init__(self,
                 min_channels,
                 max_channels,
                 in_channels,
                 block_activation=nn.ReLU(),
                 batch_norm=False,
                 drop_rate=None,
                 bias=True):
        """This is the constructor for a custom content encoder.

        Args:
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'.
            in_channels (int): Channel dimension of the input image.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(Content_Encoder, self).__init__()
        self.logger = get_logger("Content_Encoder")
        # create a list with all channel dimensions throughout the encoder.
        layers = []

        channel_numbers = [in_channels] + list(2 ** np.arange(np.log2(min_channels), np.log2(max_channels+1)).astype(np.int))
        # get all convolutional blocks with corresponding parameters
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            # add convolution
            layers.append(Conv2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias))
        # save all blocks to the class instance
        self.main = nn.Sequential(*layers)
        self.logger.debug("Content Encoder channel sizes: {}".format(channel_numbers))

    def forward(self, x):
        """This function predicts the content."""
        return self.main(x)

class Style_Encoder(nn.Module):
    def __init__(self,
                 min_channels,
                 max_channels,
                 in_channels,
                 block_activation=nn.ReLU(),
                 batch_norm=False,
                 drop_rate=None,
                 bias=True):
        """This is the constructor for a custom style encoder.

        Args:
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'.
            in_channels (int): Channel dimension of the input image.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(Style_Encoder, self).__init__()
        self.logger = get_logger("Content_Encoder")
        # create a list with all channel dimensions throughout the encoder.
        layers = []

        channel_numbers = [in_channels] + list(2 ** np.arange(np.log2(min_channels), np.log2(max_channels+1)).astype(np.int))
        # get all convolutional blocks with corresponding parameters
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            # add convolution
            layers.append(Conv2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias))
        layers.append(nn.Flatten())
        # save all blocks to the class instance
        self.main = nn.Sequential(*layers)
        self.logger.debug("Content Encoder channel sizes: {}".format(channel_numbers))

    def forward(self, x):
        """This function predicts the style."""
        return self.main(x)

    
class Decoder(nn.Module):
    """This is the decoder part of the style tranfer model."""

    def __init__(self,
                 min_channels,
                 max_channels,
                 out_channels,
                 block_activation=nn.ReLU(),
                 final_activation=nn.Tanh(),
                 batch_norm=False,
                 drop_rate=None,
                 bias=True):
        """This is the constructor for a custom decoder.

        Args:
            min_channels (int): Channel dimension before the last convolution is applied.
            max_channels (int): Channel dimension after the first convolution is applied. The channel dimension is cut in half after every convolutional block.
            out_channels (int): Channel dimension of the output image.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            final_activation (torch.nn module, optional): Activation function used in the last convolution for the output image. Defaults to nn.Tanh().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(Decoder, self).__init__()
        self.logger = get_logger("Decoder")
        # create a list with all channel dimensions throughout the decoder.
        layers = []
        channel_numbers = list(2 ** np.arange(np.log2(min_channels), np.log2(max_channels+1)).astype(np.int)[::-1]) + [out_channels]
        stride = 2
        padding = 1
        # get all convolutional blocks with corresponding parameters
        for i in range(len(channel_numbers)-1):
            activation = block_activation if i != len(channel_numbers)-1 else final_activation
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            layers.append(Transpose2dBlock(in_ch, out_ch, activation, batch_norm, drop_rate, bias, stride=stride, padding=padding))
        
        # save all blocks to the class instance
        self.main = nn.Sequential(*layers)
        self.logger.debug("Decoder channel sizes: {}".format(channel_numbers))

    def forward(self, x):
        """This function creates reconstructed image from style and content."""
        return self.main(x)

class Discriminator(nn.Module):
    """This is the discriminator of the style transfer model"""

    def __init__(self,
                out_channels,
                out_size,
                min_channels,
                max_channels,
                num_classes,
                block_activation=nn.LeakyReLU(),
                batch_norm=False,
                drop_rate=None,
                bias=True):
        """This is the constructor for the discriminator of the style transfer model

        Args:
            out_channels (int): Channel dimension of the output image.
            out_size (int): Size of the output image.
            min_channels (int): Channel dimension before the last convolution is applied.
            max_channels (int): Channel dimension after the first convolution is applied. The channel dimension is cut in half after every convolutional block.
            block_activation (torch.nn module, optional): Activation function of the convolution. Defaults to nn.LeakyReLU().
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """
        super(Discriminator, self).__init__()
        self.logger = get_logger("Discrimnator")
        conv_layers  = []
        channel_numbers = [out_channels] + list(2 ** np.arange(np.log2(min_channels), np.log2(max_channels)+1).astype(np.int))
        linear_nodes = int((out_size/2)**2 * min_channels * (1/2)**((len(channel_numbers)-2)))
        for i in range(len(channel_numbers)-1):
            in_ch = channel_numbers[i]
            out_ch = channel_numbers[i+1]
            # add convolution
            conv_layers.append(Conv2dBlock(in_ch, out_ch, block_activation, batch_norm, drop_rate, bias))
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)
        
        lin_layers = []
        
        lin_layers.append(nn.Linear(linear_nodes + num_classes, 1))
        lin_layers.append(nn.Sigmoid())

        self.lin = nn.Sequential(*lin_layers)

        # save all blocks to the class instance
        self.logger.debug("Discriminator channel sizes: {}".format(channel_numbers))
        self.logger.debug("Linear layer size: {}".format(linear_nodes + num_classes))
    
    def forward(self, x, style_labels):
        x = self.conv(x)
        style_labels = style_labels[:] * torch.ones((x.shape[0], 1)).to(x.device)
        x = torch.cat((x, style_labels), dim=1)
        return self.lin(x)

class Style_Transfer_Model_edflow(Style_Transfer_Model):
    def __init__(self, config):
        
        """This is the constructor for the full style transfer model with costum style and content encoder and decoder.

        Args:
            min_channels (int): Channel dimension after the first convolution is applied.
            max_channels (int): Channel dimension is double after every convolutional block up to the value 'max_channels'.
            in_channels (int): Channel dimension of the input image.
            out_channels (int): Channel dimension of the output image.
            block_activation (torch.nn module, optional): Activation function used in the convolutional blocks. Defaults to nn.ReLU().
            final_activation (torch.nn module, optional): Activation function used in the last convolution for the output image. Defaults to nn.Tanh().            
            batch_norm (bool, optional): Normalize over the batch size. Defaults to False.
            drop_rate (float, optional): Dropout rate for the convolutions. Defaults to None, corresponding to no dropout.
            bias (bool, optional): If the convolutions use a bias. Defaults to True.
        """

        min_channels = config["model_config"]["min_channels"]
        max_channels = config["model_config"]["max_channels"]
        in_channels = config["model_config"]["in_channels"]
        out_channels = config["model_config"]["out_channels"]
        in_size = config["model_config"]["in_size"]
        num_classes = config["model_config"]["num_classes"]
        block_activation=nn.ReLU()
        final_activation=nn.Tanh()
        batch_norm = config["model_config"]["batch_norm"]
        drop_rate = None if config["model_config"]["drop_rate"] == "None" else config["model_config"]["drop_rate"]
        bias=config["model_config"]["bias"]

        super(Style_Transfer_Model_edflow, self).__init__(min_channels,
                                                        max_channels,
                                                        in_channels,
                                                        out_channels,
                                                        in_size,
                                                        num_classes,
                                                        block_activation,
                                                        final_activation,
                                                        batch_norm,
                                                        drop_rate,
                                                        bias)



