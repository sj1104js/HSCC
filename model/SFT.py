import torch
from black.trans import Transformer
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import cv2
import numpy as np



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]  # h：9 w:9
        x = rearrange(x, 'b c h w -> b (h w) c')  # （16，64，9，9）→（16，81，64）
        x = self.norm(x)  # 对通道做归一化
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # （16，64，9，9）
        return x


class SCA(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout=0.1):  # dim=64 num_heads=8
        super(SCA, self).__init__()
        self.num_heads = num_heads  # 8
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))  # (8,1,1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.hsi_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # bias为偏置项
        self.lidar_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.hsi_dw_conv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.lidar_dw_conv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.coeff1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.project_out = nn.Sequential(
            nn.Conv2d(dim//4, dim//8, kernel_size=1),
            nn.BatchNorm2d(dim//8),
            nn.ReLU(),
            #nn.Conv2d(dim//8, dim//8, kernel_size=1)
            )

    def forward(self, hsi, lidar):

        assert hsi.shape[-2:] == lidar.shape[-2:], "H,W of hsi and lidar must be the same."

        b, c, h, w = hsi.shape

        hsi_qkv = self.hsi_qkv(hsi)
        hsi_qkv = self.hsi_dw_conv(hsi_qkv)
        hsi_q, hsi_k, hsi_v = hsi_qkv.chunk(3, dim=1)


        lidar_qkv = self.lidar_qkv(lidar)
        lidar_qkv = self.lidar_dw_conv(lidar_qkv)
        lidar_q,lidar_k, lidar_v = lidar_qkv.chunk(3, dim=1)


        hsi_q = rearrange(hsi_q, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)  #
        hsi_k = rearrange(hsi_k, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)
        lidar_q = rearrange(lidar_q, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)
        lidar_k = rearrange(lidar_k, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)
        hsi_v = rearrange(hsi_v, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        lidar_v = rearrange(lidar_v, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)

        q1 = torch.nn.functional.normalize(hsi_q, dim=-1)
        k2 = torch.nn.functional.normalize(lidar_k, dim=-1)

        q2 = torch.nn.functional.normalize(lidar_q, dim=-1)
        k1 = torch.nn.functional.normalize(hsi_k, dim=-1)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature1
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature2

        attn =torch.cat((attn1, attn2), dim=1)
        # #
        attn=self.project_out(attn)
        # #
        attn1 =attn+attn1
        attn2 = attn+attn2
        attn1= attn1.softmax(dim=-1)
        attn2= attn2.softmax(dim=-1)
        attn1 = self.dropout(attn1)
        attn2 = self.dropout1(attn2)

        hsi_out = (attn1 @ hsi_v) + hsi_v  # hsi_out(16,8,8,81)
        lidar_out = (attn2 @ lidar_v) + lidar_v  # lidar_out(16,8,8,81)

        hsi_out = rearrange(hsi_out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w) + hsi
        lidar_out = rearrange(lidar_out, 'b head c (h w) -> b (head c) h w',
                              head=self.num_heads, h=h, w=w) + lidar
        return hsi_out, lidar_out



class SCATransBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2,patch_size=9,emb_dropout=0.1):
        super(SCATransBlock, self).__init__()

        self.norm1_1 = LayerNorm(dim)
        self.norm1_2 = LayerNorm(dim)

        self.attn = SCA(dim, num_heads)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1)
        #self.pos_embedding1 = nn.Parameter(torch.empty(1,  patch_size ** 2, dim))

        self.norm2 = LayerNorm(dim)
        #self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x, y,mask=None):  # x y


        x, y = self.attn(self.norm1_1(x), self.norm1_2(y))
        attn = torch.cat((x, y), dim=1)
        attn = self.project_out(attn)

        return attn



# class CrossAttention(nn.Module):
#     def __init__(self, dim=64, num_heads=8):
#         super(CrossAttention, self).__init__()
#
#         self.linear_q = nn.Conv2d(dim, dim, 1)
#         self.linear_k = nn.Conv2d(dim, dim, 1)
#         self.linear_v = nn.Conv2d(dim, dim, 1)
#
#         self.scale = np.power(dim, 0.5)
#
#     def forward(self, hsi, lidar):
#         assert hsi.shape[-2:] == lidar.shape[-2:], "H,W of hsi and lidar must be the same."
#         b, c, h, w = hsi.shape
#
#         q = self.linear_q(hsi)
#         k = self.linear_k(hsi)
#         v = self.linear_v(lidar)
#
#         q = rearrange(q, 'b c h w -> b c (h w)')
#         k = rearrange(k, 'b c h w -> b (h w) c')
#         v = rearrange(v, 'b c h w -> b c (h w)')
#
#         attn = (q @ k) / self.scale
#         attn = attn.softmax(-1)
#
#         out = attn @ v
#         out = rearrange(out, 'b c (h w) -> b c h w',h=h,w=w)
#         return out

# class MultipleCrossAttention(nn.Module):
#     def __init__(self, dim=64):
#         super(MultipleCrossAttention, self).__init__()
#
#         self.cross_1 = CrossAttention()
#         self.cross_2 = CrossAttention()
#         self.cross_3 = CrossAttention()
#
#     def forward(self, x, y):
#         out1 = self.cross_1(x, y)
#         out2 = self.cross_2(y, x)
#         out = self.cross_3(out1, out2)
#         return out