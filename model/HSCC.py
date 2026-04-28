import torch
import pywt
import torch.nn as nn
import math
import  numpy  as np
import torch.nn.functional as F
from mamba_ssm import Mamba
from thop import profile
from einops import rearrange
from model.SFT import SCATransBlock


class HSI_Processor(nn.Module):
    """Hyperspectral Image Processor """

    def __init__(self, band=144, pca_dim=64, embed_dim=64):
        super().__init__()
        # PCA dimensionality reduction
        self.pca = nn.Conv2d(band, pca_dim, kernel_size=1)

        # Feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 4, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(4),
            nn.GELU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(4 * band, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x):
        # PCA reduction [B,144,H,W] -> [B,30,H,W]
        # x = self.pca(x)
        # 3D convolution processing
        b, c, h, w = x.shape
        x = x.view(b, 1, c, h, w)
        x = self.conv3d(x)
        x = x.view(b, -1, h, w)
        return self.conv2d(x)


class LiDAR_Processor(nn.Module):
    """LiDAR Feature Extractor """

    def __init__(self, band=1, embed_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(band, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.feature_extractor(x)



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


class transformer(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout=0.1):  # dim=64 num_heads=8
        super(transformer, self).__init__()
        self.num_heads = num_heads  # 8
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # (8,1,1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)


        self.dropout = nn.Dropout(dropout)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        b, c, h, w = x.shape
        x_q, x_k, x_v = self.conv1(x), self.conv2(x), self.conv3(x)

        x_q = rearrange(x_q, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        x_k = rearrange(x_k, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)
        x_v = rearrange(x_v, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)

        q = torch.nn.functional.normalize(x_q, dim=-1)
        k = torch.nn.functional.normalize(x_k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x_out = (attn @ x_v) + x_v


        x_out = rearrange(x_out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w) + x

        return x_out

class EnhancedAttentionLayer(nn.Module):
    # y attends x
    def __init__(self, x_channel, y_channel, out_channel,spatial=True):
        super(EnhancedAttentionLayer, self).__init__()
        self.out_channel = out_channel


        if spatial:
            self.channel_affine = nn.Linear(x_channel, out_channel)


        self.y_affine = nn.Linear(y_channel, out_channel, bias=False)

        self.attention1 = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attention = transformer(dim=64,num_heads=8,dropout=0.1)

        self.attn_weight_affine = nn.Linear(out_channel, 1)
        self.norm1_1 = LayerNorm(64)
        self.norm1_2 = LayerNorm(64)

    def forward(self, x, y):
        # x -> B C H W
        # y -> B C H W

        x_1=x.permute(0,2,3,1) # x_1 b h w c
        y_1=y.permute(0,2,3,1) # x_2 b h w c
        # print(x.size(), y.size())

        x_attn = self.attention(x) # x_attn B C H W
        x_tensor = x_attn.permute(0, 2, 3, 1)  # x_tensor B H W C

        y_attn = self.attention(y) # y_attn B C H W
        y_tensor = y_attn.permute(0, 2, 3, 1) # y_tensor B H W C
        y_k = self.y_affine(y_tensor) # (B H W C)
        x_k = self.channel_affine(x_tensor) # (B H W C)

        x_k += y_k
        x_k = torch.tanh(x_k)
        x_attn_weights = self.attn_weight_affine(x_k).squeeze(-1) # (S_v, H_v, W_v)
        spatial_attn_weights=F.softmax(x_attn_weights.reshape(x_tensor.size(0), -1),dim=-1).reshape(x_tensor.size(0), x_tensor.size(1), x_tensor.size(2)) # (S_v, H_v, W_v)
        weight=spatial_attn_weights.unsqueeze(-1)

        result1 = x_1*weight
        result1 += x_1
        result1=result1.permute(0,3,1,2) # b c h w

        result2 = y_1*weight
        result2 += y_1
        result2=result2.permute(0,3,1,2) # b c h w
        return result1, result2


class HGM(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.attn1 = SCATransBlock()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hsi_feat, lidar_feat):

        att_loss, out1 = self.attn1(hsi_feat,lidar_feat)

        hsi_feat = hsi_feat + out1
        lidar_feat = lidar_feat + out1

        return hsi_feat, lidar_feat, out1
        #return hsi_feat,lidar_feat


class HSCC(nn.Module):
    """Complete Network Architecture"""

    def __init__(self, hsi_bands, lidar_bands, num_classes=15):
        super().__init__()
        self.hsi_net = HSI_Processor(hsi_bands)
        self.lidar_net = LiDAR_Processor(lidar_bands)
        self.fusion = HGM()


        # Learnable fusion coefficients
        self.coeff1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff3 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff4 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff5 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff6 = torch.nn.Parameter(torch.Tensor([0.5]))


        self.refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        self.output_norm = nn.BatchNorm1d(num_classes)


        self.classifier1 = nn.Sequential(
            nn.Linear(64, 32), #（64，32）
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes) #（64，11）
        )


        self._init_weights()

        self.cmsa = EnhancedAttentionLayer(x_channel=64, y_channel=64, out_channel=64, spatial=True)

        mlp_dim1=64
        mlp_dim2=mlp_dim1*2
        self.criterion = nn.CosineSimilarity(dim=1)

        #self.fc = nn.Linear(384, num_classes)
        self.fc1 = nn.Linear(64, num_classes)
        self.classifier_mlp = nn.Sequential(
            nn.Linear(mlp_dim1, mlp_dim2),
            nn.BatchNorm1d(mlp_dim2),
            # nn.Dropout(0.1),
            nn.ReLU(),
            # nn.Linear(mlp_dim2, mlp_dim2),
            # nn.BatchNorm1d(mlp_dim2),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(mlp_dim2, num_classes),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, hsi, lidar):

        # Feature extraction
        hsi_feat=self.hsi_net(hsi)
        b, c, h, w = hsi_feat.shape
        lidar_feat=self.lidar_net(lidar)
        data11 = hsi_feat
        data22 = lidar_feat
        f_H0 = data11.view(b, 64, h*w)
        f_H0 = f_H0.permute(0, 2, 1) # B HW C
        f_H0 = torch.mean(f_H0, dim=1)
        f_H0 = self.classifier_mlp(f_H0) #1
        f_L0 = data22.view(b, 64, h*w)
        f_L0 = f_L0.permute(0, 2, 1)
        f_L0 = torch.mean(f_L0, dim=1)
        f_L0 = self.classifier_mlp(f_L0) #1


        hsi_feat1,lidar_feat1 = self.cmsa(hsi_feat, lidar_feat)

        x11 = hsi_feat1.view(b, 64, h*w)
        x12 = x11.permute(0, 2, 1)
        x21 = lidar_feat1.view(b, 64, h*w)
        x22 = x21.permute(0, 2, 1)
        f_H = torch.mean(x12, dim=1)
        f_H = self.classifier_mlp(f_H)
        f_L = torch.mean(x22, dim=1)
        f_L = self.classifier_mlp(f_L)



        hsi_feat2,lidar_feat2 ,fuse= self.fusion(hsi_feat1, lidar_feat1)


        hsi_feat22=self.refine (hsi_feat2).flatten(1)
        hsi_feat22=self.classifier1(hsi_feat22)
        lidar_feat22=self.refine (lidar_feat2).flatten(1)
        lidar_feat22=self.classifier1(lidar_feat22)

        sim1 = self.criterion(f_H0, f_L0)
        sim2 = self.criterion(f_H, f_L)
        sim3 = self.criterion(hsi_feat22, lidar_feat22)
        sim4 = self.criterion(hsi_feat22, f_L0)
        sim5 = self.criterion(hsi_feat22, f_L)
        sim6 = self.criterion(lidar_feat22, f_H0)
        sim7 = self.criterion(lidar_feat22, f_H)
        loss1=(((-(sim1.mean() + sim2.mean()+sim3.mean())) + 3)) * 0.1
        loss2 = (((-(sim4.mean() + sim5.mean())) + 2)) * 0.1
        loss3 = (((-(sim6.mean() + sim7.mean())) + 2)) * 0.1
        #consistency_vec = torch.stack([sim1,sim2,sim3,sim4,sim5,sim6,sim7], dim=1)
        scc_loss=loss1+loss2+loss3

        out1=self.coeff1*hsi_feat+self.coeff2*lidar_feat+self.coeff3*hsi_feat1+self.coeff4*hsi_feat2+self.coeff5*lidar_feat2+self.coeff6*lidar_feat1
        #out1 =  hsi_feat +  lidar_feat +  hsi_feat1 +  hsi_feat2 +  lidar_feat2 + lidar_feat1
        out=self.avg_pool(out1)
        out = torch.flatten(out, 1)
        out=self.fc1(out)
        out = self.output_norm(out)
        out = F.softmax(out, dim=1)

        return scc_loss,out
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



