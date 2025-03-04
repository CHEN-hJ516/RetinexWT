import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 通道上取最大值 通道维数-->1
        avg_pool = torch.mean(x, 1, keepdim=True)  # 通道上取平均值 通道维数-->1
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SCAM(nn.Module):
    def __init__(self, f_number, excitation_factor=2, kernel_size=5, reduction=16, bias=False):
        super(SCAM, self).__init__()
        # Initialize MLP, SALayer, and CALayer
        self.mlp = MLP(f_number, excitation_factor)
        self.sa_layer = SALayer(kernel_size, bias)
        self.ca_layer = CALayer(f_number, reduction, bias)

    def forward(self, x):
        # Pass through MLP
        x = self.mlp(x)
        # Pass through SALayer
        x = self.sa_layer(x)
        # Pass through CALayer
        x = self.ca_layer(x)
        return x

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1,32, 64, 64)
    # 创建一个实例
    scam = SCAM(32, reduction=16, bias=True)
    # 将输入特征图传递给模块
    output = scam(input)
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output_L shape: {output.shape}")