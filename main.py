import torch
from torch import nn
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import functional as F
import math
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1,2))
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),

        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)
        self.thresh=nn.Parameter(torch.randn(1))
    def forward(self, x):
        block=1
        if self.shifted:
           block=2
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class CConvFuUnit(nn.Module):
    """
    A Convolutional Block with depthwise separable convolutions followed by
    instance normalization, LeakyReLU activation, and optional dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, kernel_size=3, stride=1, padding=1, drop_prob=0):
        """
        Args:
            in_chans: Number of input channels.
            out_chans: Number of output channels.
            kernel_size: Size of the convolutional kernel. Default: 3.
            stride: Convolution stride. Default: 1.
            padding: Padding added to input tensor. Default: 1.
            drop_prob: Dropout probability. Default: 0 (no dropout).
        """
        super(CConvFuUnit, self).__init__()

        self.H1layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, groups=in_chans),
            nn.InstanceNorm2d(in_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_chans, (7 * in_chans) // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d((7 * in_chans) // 8),
        )

        self.H1layersdepth = nn.Sequential(
            nn.Conv2d(in_chans, (7 * in_chans) // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d((7 * in_chans) // 8),
        )

        self.H2layers = nn.Sequential(
            nn.Conv2d((7 * in_chans) // 8, (7 * in_chans) // 8, kernel_size=3, stride=1, padding=1, groups=(7 * in_chans) // 8),
            nn.InstanceNorm2d((7 * in_chans) // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((7 * in_chans) // 8, (3 * in_chans) // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d((3 * in_chans) // 4),
        )

        self.H3layers = nn.Sequential(
            nn.Conv2d((3 * in_chans) // 4, (3 * in_chans) // 4, kernel_size=3, stride=1, padding=1, groups=(3 * in_chans) // 4),
            nn.InstanceNorm2d((3 * in_chans) // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((3 * in_chans) // 4, (5 * in_chans) // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d((5 * in_chans) // 4),
        )

        self.H4layers = nn.Sequential(
            nn.Conv2d((5 * in_chans) // 4, (5 * in_chans) // 4, kernel_size=3, stride=1, padding=1, groups=(5 * in_chans) // 4),
            nn.InstanceNorm2d((5 * in_chans) // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((5 * in_chans) // 4, out_chans, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_chans),
        )

        self.L1layers = nn.Sequential(
            nn.Conv2d(in_chans, (7 * in_chans) // 8, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d((7 * in_chans) // 8),
        )

        self.L2layers = nn.Sequential(
            nn.Conv2d((7 * in_chans) // 8, (3 * in_chans) // 4, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d((3 * in_chans) // 4),
        )

        self.L3layers = nn.Sequential(
            nn.Conv2d((3 * in_chans) // 4, (5 * in_chans) // 4, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d((5 * in_chans) // 4),
        )

        self.L4layers = nn.Sequential(
            nn.Conv2d((5 * in_chans) // 4, out_chans, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_chans),
        )

        self.skip_1 = nn.Sequential(
            nn.Conv2d(in_chans, (3 * in_chans) // 4, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d((3 * in_chans) // 4),
        )

        self.skip_2 = nn.Sequential(
            nn.Conv2d(in_chans, (5 * in_chans) // 4, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d((5 * in_chans) // 4),
        )

        self.skip_final = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_chans),
        )

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CConvFuUnit.

        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        image_H1 = self.H1layers(image)
        image_L1 = self.L1layers(image)
        image_HL = self.relu(image_H1 + image_L1)

        image_H2 = self.H2layers(image_HL)
        image_L2 = self.L2layers(image_HL)
        image_skip = self.skip_1(image)
        image_HL = self.relu(image_H2 + image_L2 + image_skip)

        image_H3 = self.H3layers(image_HL)
        image_L3 = self.L3layers(image_HL)
        image_skip = self.skip_2(image)
        image_HL = self.relu(image_H3 + image_L3 + image_skip)

        image_H4 = self.H4layers(image_HL)
        image_L4 = self.L4layers(image_HL)
        image_skip = self.skip_final(image)

        return self.relu(image_H4 + image_L4 + image_skip)


class ConvFuUnit(nn.Module):
    """
    A Hybrid Convolutional Block combining depthwise and pointwise convolutions
    with instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int, kernel_size=3, stride=1, padding=1, drop_prob=0):
        """
        Args:
            in_chans: Number of input channels.
            out_chans: Number of output channels.
            kernel_size: Size of the convolutional kernel. Default: 3.
            stride: Convolution stride. Default: 1.
            padding: Padding added to input tensor. Default: 1.
            drop_prob: Dropout probability. Default: 0 (no dropout).
        """
        super(ConvFuUnit, self).__init__()

        self.convH3 = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, groups=in_chans)
        self.InstH1 = nn.InstanceNorm2d(in_chans)
        self.convH1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.InstH2 = nn.InstanceNorm2d(out_chans)

        self.convL1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.InstL2 = nn.InstanceNorm2d(out_chans)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConvFuUnit.

        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        image_H = self.convH3(image)
        image_H = self.InstH1(image_H)
        image_H = self.relu(image_H)
        image_H = self.convH1(image_H)
        image_H = self.InstH2(image_H)

        image_L = self.convL1(image)
        image_L = self.InstL2(image_L)

        return self.relu(image_H + image_L)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        # super().__init__()
        super(ConvBlock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int,kernel_size=2, stride=2,padding=0):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size, stride, padding,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)



class Decoder(nn.Module):
    def __init__(self, in_chans=768*2, out_chans=1, chans=768, num_pool_layers=4):
        super(Decoder, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = 0.0

        # Define convolutional layers
        self.conv = ConvFuUnit(self.chans, self.chans * 2)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_head = nn.ModuleList()

        self.chans *= 2

        for idx in range(num_pool_layers + 2):
            if idx % 2 == 0:
                self.up_transpose_conv.append(nn.Sequential(TransposeConvBlock(self.chans, self.chans // 2)))
                self.up_conv.append(nn.Sequential(ConvFuUnit(self.chans, self.chans // 2)))
            else:
                self.up_transpose_conv.append(nn.Sequential(TransposeConvBlock(self.chans, self.chans // 2)))
                self.up_conv.append(nn.Sequential(CConvFuUnit(self.chans, self.chans // 2)))

            self.chans //= 2

        self.up_head.append(nn.Sequential(
            ConvFuUnit(self.chans, self.chans // 2),
            nn.Conv2d(self.chans // 2, self.out_chans, kernel_size=1, stride=1, padding=0),
        ))

    def forward(self, swin_enc_features):
        output = self.conv(F.avg_pool2d(swin_enc_features[-1], 2))

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            output = transpose_conv(output)
            if swin_enc_features:
                output = torch.cat((swin_enc_features.pop(), output), dim=1)
                output = conv(output)

        output = self.up_head[0](output)
        return output

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)

        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class HyCoSwin(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=1, head_dim=48, window_size=5,
                 downscaling_factors=(4,2, 2, 2, 2), relative_pos_embedding=True):
        super(HyCoSwin,self).__init__()

        self.initial_layer1=ConvBlock(1, hidden_dim//4,drop_prob=0)
        self.initial_layer2=ConvBlock(1, hidden_dim//2,drop_prob=0)

        self.patches1=PatchMerging(in_channels=channels, out_channels=hidden_dim//4, downscaling_factor=1)
        self.patches2=PatchMerging(in_channels=channels, out_channels=hidden_dim//2, downscaling_factor=2)

        self.stage0 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage1 = StageModule(in_channels=hidden_dim, hidden_dimension=2*hidden_dim, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.Decoder=Decoder(in_chans=hidden_dim * 16, out_chans=1,chans=hidden_dim * 8)

    def forward(self, img):

        swin_enc_features=[]
        patch1=self.initial_layer1(img)
        patch2=self.initial_layer2(img)
        patch2=F.avg_pool2d(patch2,2)

        swin_enc_features.append(patch1)
        swin_enc_features.append(patch2)

        x0 = self.stage0(img)
        swin_enc_features.append(x0)
        x1 = self.stage1(x0)
        swin_enc_features.append(x1)
        x2 = self.stage2(x1)
        swin_enc_features.append(x2)
        x3 = self.stage3(x2)
        swin_enc_features.append(x3)

        x = self.Decoder(swin_enc_features)

        return x



def HyCoSwin_Model(hidden_dim=144, layers=(2,2, 2,2, 2), heads=(6, 12, 24, 48), **kwargs):
    return HyCoSwin(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

if __name__=='__main__':
    x=torch.rand((1,1,320,320))
    print(x.shape)
    model=HyCoSwin_Model()
    y=model(x)
    print('y :',y.shape)
