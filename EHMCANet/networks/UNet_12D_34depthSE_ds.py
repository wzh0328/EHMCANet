import torch.nn as nn
import torch
import Config as config
config_vit = config.get_CTranS_config()
from thop import profile
# from tensorflow.keras.layers import concatenate
import torch.nn.functional as F


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UpBlock_avg(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock_avg, self).__init__()
        # upsampled_tensor = F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.up = nn.Upsample(scale_factor=16)   ###img_szie=256时
        # self.up = nn.Upsample(scale_factor=14)   ###img_szie=224时
        # self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        # print('out.shape',out.shape)
        # print('skip_x.shape',skip_x.shape)

        ####img_szie=256时
        # out.shape
        # torch.Size([16, 512, 16, 16])
        # skip_x.shape
        # torch.Size([16, 512, 14, 14])

        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)




#低级细节特征自注意机制
class DFMAS(nn.Module):
    def __init__(self, in_ch):
        super(DFMAS, self).__init__()
        self.in_ch3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.in_ch5 = nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=True)
        self.in_ch7 = nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.out_ch = nn.Conv2d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)
        # self.cSE = cSE(in_ch)


    def forward(self, x):
        o1 = self.in_ch3(x)
        o2 = self.in_ch5(x)
        o3 = self.in_ch7(x)
        out = o1 + o2 + o3
        # out = concatenate([o1, o2, o3])
        # out = cSE(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.out_ch(out)
        out = self.sigmoid(out)
        out_ch = x*out

        # b,c,h,w = out_ch.shape
        # out_ch = torch.ones(b,c,h,w)
        # cSE = cSE()
        # out_ch = cSE(out_ch)
        return out_ch


# Original sSE、cSE、scSE block
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        # self.Conv1x1 = DepthwiseSeparableConv1(in_channels, 1, kernel_size=1, stride=1,padding=0)   ####深度可分离卷积
        self.Conv1x1 = DepthWiseConv2d(in_channels, 1, kernel_size=1, stride=1,padding=0)   ####深度可分离卷积
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Excitation = DepthWiseConv2d(in_channels//2, in_channels, kernel_size=1, stride=1,padding=0)
        self.norm = nn.Sigmoid()
        self.Conv_Squeeze = DepthWiseConv2d(in_channels, in_channels // 2, kernel_size=1, stride=1,padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        # z = self.relu(z)
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)


class DepthwiseSeparableConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv1, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


# class scSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.cSE = cSE(in_channels)
#         self.sSE = sSE(in_channels)
#         self.conv_concat = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)  # 1x1 convolution after concatenation
#
#     def forward(self, U):
#         U_sse = self.sSE(U)
#         U_cse = self.cSE(U)
#
#         # Concatenate the output of cSE and sSE along the channel dimension (dim=1)
#         U_concat = torch.cat((U_cse, U_sse), dim=1)
#
#         # Apply a 1x1 convolution on the concatenated feature maps
#         U_combined = self.conv_concat(U_concat)
#
#         # return U_cse+U_sse
#         return U_combined


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)
        # self.depthwise_conv = DepthwiseSeparableConv3(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = DepthWiseConv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)   ####Test_session_04.01_15h13

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)

        # Concatenate U_cse and U_sse along the channel dimension
        U_concat = torch.cat((U_cse, U_sse), dim=1)

        # Apply depthwise separable convolution
        output = self.depthwise_conv(U_concat)

        output = output + U   ####加上残差   Test_session_04.01_15h30。MoNuSeg的Dice=80.74

        return output


class gt_UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(gt_UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2, stride=2)
        # self.up = nn.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    # def forward(self, x, skip_x):
    def forward(self, x):
        out = self.up(x)
        # x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(out)


def upsample_bilinear(input_tensor, scale_factor):
    """
    使用双线性插值上采样输入张量。

    Parameters:
    - input_tensor (torch.Tensor): 输入张量
    - scale_factor (float): 上采样的比例因子

    Returns:
    - torch.Tensor: 上采样后的张量
    """
    # 使用双线性插值上采样
    upsampled_tensor = F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    return upsampled_tensor


class UNet_12D_34depthSE_ds(nn.Module):
    ######把cSE和sSE中1*1CONV，换成深度可分离卷积，还有把Concat之后的卷积也换成深度可分离卷积
    def __init__(self,config, n_channels=3, n_classes=9,vis=False,gt_ds=True):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()

        self.vis = vis
        self.gt_ds = gt_ds

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        # in_channels = 64
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)


        ###加入DFMAS模块
        self.DF1 = DFMAS(in_channels)
        # self.scSE1 = scSE(in_channels)
        # self.DFu2 = DFMAS(in_channels)
        self.DF2 = DFMAS(in_channels*2)
        # self.scSE2 = scSE(in_channels*2)

        # Original sSE、cSE、scSE block
        self.scSE3 = scSE(in_channels*4)
        self.scSE4 = scSE(in_channels*8)
        self.scSE_up3 = scSE(in_channels*2)

        # An Improved sSE、cSE、scSE block
        # self.scSE3 = ChannelSpatialSqueezeExcitation(in_channels*4)
        # self.scSE4 = ChannelSpatialSqueezeExcitation(in_channels*8)



        self.up5 = UpBlock_avg(in_channels*16, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))



        if gt_ds:
            print('gt deep supervision was used')
            ##第四层经过三次转置卷积
            self.ds_0up4 = gt_UpBlock(in_channels*4,in_channels*2,nb_Conv=2)
            self.ds_1up4 = gt_UpBlock(in_channels*2,in_channels*1,nb_Conv=2)
            self.ds_2up4 = gt_UpBlock(in_channels,n_classes,nb_Conv=2)
            ##第三层经过两次次转置卷积
            self.ds_0up3 = gt_UpBlock(in_channels*2,in_channels,nb_Conv=2)
            self.ds_1up3 = gt_UpBlock(in_channels,n_classes,nb_Conv=2)
            ##第二层经过一次转置卷积
            self.ds_up2 = gt_UpBlock(in_channels ,n_classes,nb_Conv=2)
            self.ds_up1 = gt_UpBlock(in_channels ,n_classes,nb_Conv=2)
            # 第一层不用上采样,但是需要把通道数从64变成1
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels, n_classes, 1))


        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        # x = x.float()
        ###在编码器的12层加D模块，在编码器的34层加scSE模块，并且scSE在34层解码器中也有
        x1 = self.inc(x)
        x1_0 = self.DF1(x1)

        x2 = self.down1(x1_0)
        # x2 = self.down1(x1)
        x2_0 = self.DF2(x2)

        x3 = self.down2(x2_0)
        # x3 = self.down2(x2)
        x3_1 = self.scSE3(x3)

        x4 = self.down3(x3_1)
        # x4 = self.down3(x3)
        x4_1 = self.scSE4(x4)

        x5 = self.down4(x4_1)
        # x5 = self.down4(x4)

        avg = self.avgpool(x5)  # shape: [bs, c, h, w] to [bs, c, 1, 1]

        x6 = self.up5(avg,x5)
        x_u4 = self.up4(x6, x4_1)

        ###把scSE模块加入解码器中
        x_u4 = self.scSE3(x_u4)
        x_u3 = self.up3(x_u4, x3_1)
        x_u3 = self.scSE_up3(x_u3)
        x_u2 = self.up2(x_u3, x2_0)
        x_u1 = self.up1(x_u2, x1_0)





        if self.gt_ds:
            ####把解码器输出变成和GT一样的大小
            #第四层经过三次上采样
            gt_0pre4 = self.ds_0up4(x_u4)
            gt_1pre4 = self.ds_1up4(gt_0pre4)
            gt_2pre4 = self.ds_2up4(gt_1pre4)
            #第三层经过两次
            gt_0pre3 = self.ds_0up3(x_u3)
            gt_1pre3 = self.ds_1up3(gt_0pre3)
            #第二层上采样
            gt_pre2 = self.ds_up2(x_u2)
            # 第一层不用上采样,但是需要把通道数从64变成1
            x_0u1 = self.gt_conv4(x_u1)
            # x_0u1 = self.outc(x_u1) #####20240423后更改


        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x_u1))
            # print()
        else:
            logits = self.outc(x_u1)
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        # return logits
        if self.gt_ds:
            # return torch.sigmoid(gt_2pre4), torch.sigmoid(gt_1pre3), torch.sigmoid(gt_pre2), torch.sigmoid(x_0u1), logits
            # return torch.sigmoid(gt_2pre4), torch.sigmoid(gt_1pre3), torch.sigmoid(gt_pre2) , logits
            # return torch.sigmoid(gt_2pre4), torch.sigmoid(gt_1pre3), torch.sigmoid(gt_pre2) ,logits
            # return torch.sigmoid(x_u4), torch.sigmoid(x_u3), torch.sigmoid(x_u2) ,logits
            # return gt_2pre4, gt_1pre3, gt_pre2,logits   ###输出热力图
            # return x_u4, x_u3, x_u2 ,logits  ###输出底色是蓝绿色的热力图
            # return x1,x2, x3,x4
            # return x1,x1_0,x2,x2_0   ###编码器输出
            # return x_u4,x_u3,x_u2,x_u1 ##解码器输出，没有经过上采样
            # return x3,x3_1,x4,x4_1
            # return x5,avg,x5,avg
            # return torch.sigmoid(x1),torch.sigmoid(x1_0), torch.sigmoid(x2),torch.sigmoid(x2_0)
            # return x4_1,x3_1, x2_0,x1_0    ###编码器输出热力图
            return x_u4, x_u3, x_u2, x_u1    ###解码器输出热力图

        else:
            return logits


if __name__ == '__main__':
    x = torch.randn(1, 3,256, 256).cuda()
    net = UNet_12D_34depthSE_ds(config_vit,n_channels=config.n_channels,n_classes=config.n_labels).cuda()
    y = net(x)

    flops, params = profile(net.cuda(), inputs=(x,))
    print("参数量：", params / 1e6)
    print("FLOPS：", flops / 1e9)

