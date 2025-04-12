import torch
from torch import nn
import torch.nn.functional as F
import math
from util import Smooth, EqualizedLinear
from typing import Tuple, Optional

class Generator(nn.Module):
    def __init__(self, log_res, dims_w, num_features=32, max_features = 512):
        super().__init__()
        #log_res = math.log2(res)
        self.args = [log_res, dims_w, num_features, max_features]
        features = [min(max_features, num_features* (2**i)) for i in range(log_res-2, -1, -1)]
        assert len(features) == (log_res-1)
        self.num_blocks = log_res-1
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(dims_w, features[0], features[0])
        self.to_rgb = ToRGB(dims_w, features[0])

        blocks = [GeneratorBlock(dims_w, features[i-1], features[i]) for i in range(1, self.num_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.up_sample = UpSample()
        print()

    def forward(self, w, input_noise):
        batch_size = w.shape[1]
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.num_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i-1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new
        return rgb
class GeneratorBlock(nn.Module):
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        # Second [style block](#style_block)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        # *toRGB* layer
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])

        # Get RGB image
        rgb = self.to_rgb(x, w)

        # Return feature map and rgb image
        return x, rgb
    
class ToRGB(nn.Module):
    def __init__(self, d_latent: int, features: int):
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])

class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        return self.smooth(self.up_sample(x))

class StyleBlock(nn.Module):

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])
    
class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        #self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.weight = nn.Parameter(torch.randn(out_features, in_features, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_features * kernel_size * kernel_size)
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]

        weights = self.weight * self.scale
        # Get [learning rate equalized weights](#equalized_weight)
        weights = weights[None, :, :, :, :] * s

        # Demodulate
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)