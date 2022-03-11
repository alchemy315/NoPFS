"""Main model architecture for CosmoFlow."""

import functools
import operator
import torch
import torch.nn


class CosmoFlowConvBlock(torch.nn.Module):
    """Convolution block for CosmoFlow."""

    def __init__(self, input_channels, output_channels, kernel_size,
                 act, pool, padding='valid'):
        """Set up a CosmoFlow convolutional block.

        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        act: Activation function.
        pool: Pooling function.
        padding: Type of padding to apply, either 'valid' or 'same'.

        """
        super().__init__()
        # Compute padding.
        self.pad_layer = None
        if padding == 'valid':
            pad_size = 0
        elif padding == 'same':
            # Same padding is easy for odd kernel sizes.
            if kernel_size % 2 == 1:
                pad_size = (kernel_size - 1) // 2
            else:
                # For even kernel sizes, we have to manually pad because we
                # need different padding on each side.
                # We follow the TF/Keras convention of putting extra padding on
                # the right and bottom.
                pad_size = 0
                kernel_sizes = [kernel_size]*3
                tf_pad = functools.reduce(
                    operator.__add__,
                    [(k // 2 + (k - 2*(k//2)) - 1, k // 2) for k
                     in kernel_sizes[::-1]])
                self.pad_layer = torch.nn.ConstantPad3d(tf_pad, value=0.0)
        else:
            raise ValueError(f'Unknown padding type {padding}')
        self.conv = torch.nn.Conv3d(input_channels, output_channels,
                                    kernel_size, padding=pad_size)
        self.act = act()
        self.pool = pool(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        return self.pool(self.act(self.conv(x)))


class CosmoFlowModel(torch.nn.Module):
    """Main CosmoFlow model."""

    def __init__(self, input_shape, output_shape,
                 conv_channels=16, kernel_size=2, n_conv_layers=5,
                 fc1_size=128, fc2_size=64,
                 act=torch.nn.LeakyReLU,
                 pool=torch.nn.MaxPool3d,
                 dropout=0.0):
        """Set up the CosmoFlow model.

        input_channels: Dimensions of the input (excluding batch).
            This should be (channels, height, width, depth).
        output_shape: Dimensions of the output (excluding batch).
            This should be the number of regression targets.
        conv_channels: Number of channels in the first conv layer.
            This will increase by a factor of 2 for enach layer.
        kernel_size: Convolution kernel size.
        n_conv_layers: Number of convolutional blocks.
        fc1_size: Number of neurons in the first fully-connected layer.
        fc2_size: Number of neurons in the second fully-connected layer.
        act: Activation function.
        pool: Pooling function.
        dropout: Dropout rate.

        """
        super().__init__()
        # Build the convolutional stack.
        conv_layers = []
        conv_layers.append(CosmoFlowConvBlock(
            input_shape[0], conv_channels, kernel_size, act, pool,
            padding='same'))
        for i in range(1, n_conv_layers):
            conv_layers.append(CosmoFlowConvBlock(
                conv_channels * 2**(i-1), conv_channels * 2**i, kernel_size,
                act, pool, padding='same'))

        # Compute output height/width/depth/channels for first FC layer.
        out_channels = conv_channels * 2**(n_conv_layers-1)
        out_height = input_shape[1] // 2**n_conv_layers
        out_width = input_shape[2] // 2**n_conv_layers
        out_depth = input_shape[3] // 2**n_conv_layers

        # Build the FC stack.
        fc_layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(
                out_channels*out_height*out_depth*out_width, fc1_size),
            act(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(fc1_size, fc2_size),
            act(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(fc2_size, output_shape),
            torch.nn.Tanh()
        ]

        self.layers = torch.nn.Sequential(*conv_layers, *fc_layers)

    def forward(self, x):
        # Output is scaled by 1.2.
        return self.layers(x) * 1.2
