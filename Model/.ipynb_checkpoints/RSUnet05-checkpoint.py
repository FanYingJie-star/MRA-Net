import torch
import torch.nn as nn
from timm.models.layers import  trunc_normal_

import torch.nn.functional as F
import math

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm = True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Batch_Norm:
          self.Conv_forward = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x

class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(single_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class instanceSegmenationHead(nn.Module):
    def __init__(self, in_chans=54, out_chans=36):
        super(instanceSegmenationHead, self).__init__()
        self.ins_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, 0))

    def forward(self, y):

        x = self.ins_conv(y)
        return x


class InstanceCounter(nn.Module):
    def __init__(self, input_n_filters, out, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters
        self.n_filters = out
        self.out_filter = out
        self.usegpu = usegpu
        self.output = nn.Sequential(nn.Linear(self.out_filter, 1),
                                    nn.Sigmoid())
        self.cnn = nn.Sequential(
                                 single_conv_relu_batch(self.input_n_filters, self.n_filters, kernel_size=3, padding=1),
                                 nn.Conv2d(self.n_filters, self.out_filter, 1, 1, 0),
                                 nn.AdaptiveAvgPool2d((1, 1))
                                 )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x

class feHead(nn.Module):
    def __init__(self, in_chs=64, out_chs=32):
        super(feHead, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_chs),
                                  nn.ReLU(inplace=True))
        self.ins_head = instanceSegmenationHead(in_chans=out_chs, out_chans=10)
        self.seg_head = instanceSegmenationHead(in_chans=out_chs, out_chans=2)
        self.counter = InstanceCounter(input_n_filters=out_chs, out=10)
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        ins_head = self.ins_head(x)
        seg_head = self.seg_head(x)
        count = self.counter(x)
        return seg_head, ins_head, count
class singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(singleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class singleConv0(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(singleConv0, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(single_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(double_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            single_conv_relu_batch(in_ch, out_ch, kernel_size=1, stride=1, padding=0, conv_op=conv_op),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class doubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


# 通道注意力
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_chs, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attn = channel_attention(in_chs=in_chs, ration=ratio)
        self.spatial_attn = spatial_attention(kernel_size=kernel_size)

    def forward(self, inputs):

        x = self.channel_attn(inputs)
        x = self.spatial_attn(x)

        return x
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class cross_conv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_op=nn.Conv2d):
        super(cross_conv, self).__init__()
         #32, 64, 128, 256
        self.conv0 = singleConv0(in_ch, out_ch, kernel_size=1, stride=2, padding=0, conv_op=nn.Conv2d) # 降维

        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(singleConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d),
                                   singleConv(out_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d)) # 降维


        self.conv3 = singleConv(in_ch, out_ch, kernel_size=2, stride=2, padding=0, conv_op=conv_op)
        self.conv3_1= singleConv(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), conv_op=conv_op)
        self.conv3_2 = singleConv(out_ch, out_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), conv_op=conv_op)
        
        self.sa = SpatialAttentionModule()
        self.eca = ECA(in_channel = out_ch )

        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

        self.conv_block1 = nn.Sequential(nn.Conv2d(out_ch*3, in_ch, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(in_ch),
                                   nn.ReLU(inplace=True))
        
        self.lga2 = LocalGlobalAttention(output_dim=out_ch,patch_size=2)
        self.lga4 = LocalGlobalAttention(output_dim=out_ch,patch_size=4)
        
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch))


        self.conv_block3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_ch)
                                    )
        self.conv1x1 = singleConv0(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        g = self.conv0(x)#[16, out_ch, 160, 80]

        y = self.maxpool(x)
        y = self.conv1(y)  #[16, out_ch, 160, 80]

        k = self.conv3(x)
        k1 = self.conv3_1(k)  #[16, out_ch, 160, 80]
        k2 = self.conv3_2(k)  #[16, out_ch, 160, 80]

        x = torch.cat([y, k1, k2], dim=1)  #[16, out_ch*3, 160, 80]
        #x = self.channnel_attn(x) * x        
        f = x
        x = self.conv_block1(x)
        x_lga2 = self.lga2(x)
        x_lga4 = self.lga4(x)
        x = self.conv_block2(x)
        x = x_lga2 + x_lga4 + x

        f = self.conv1x1(f)
        x = self.conv_block3(x)
        
        x = self.eca(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.relu(x+f)

        return x


class RsUnet05(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(RsUnet05, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Batch_Norm = True)
 
        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Batch_Norm = True)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Batch_Norm = True)
 
        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Batch_Norm = True)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Batch_Norm = True)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Batch_Norm = True)
 
        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Batch_Norm = True)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Batch_Norm = True)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Batch_Norm = True)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm = True)
 
 
        self.stage_0 = singleConv(3, 64, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d)
        self.stage_1 = cross_conv(64, 128, conv_op=nn.Conv2d)
        self.stage_2 = cross_conv(128, 256, conv_op=nn.Conv2d)
        self.stage_3 = cross_conv(256, 512, conv_op=nn.Conv2d)
        self.stage_4 = cross_conv(512, 1024, conv_op=nn.Conv2d)
 
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = feHead(in_chs=64, out_chs=num_classes)

        self.final_super_0_2 = feHead(in_chs=64, out_chs=num_classes)
        self.final_super_0_3 = feHead(in_chs=64, out_chs=num_classes)
        self.final_super_0_4 = feHead(in_chs=64, out_chs=num_classes)
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(x_0_0)
        x_2_0 = self.stage_2(x_1_0)
        x_3_0 = self.stage_3(x_2_0)
        x_4_0 = self.stage_4(x_3_0)
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            seg_head, ins_head, count = self.final_super_0_4(x_0_4)
            return seg_head, ins_head, count
 
 
if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    model = RsUnet05(num_classes=12, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)    
    
    print("deep_supervision: True")
    deep_supervision = True
    model = RsUnet05(num_classes=12, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
      print(out.shape)
 
 