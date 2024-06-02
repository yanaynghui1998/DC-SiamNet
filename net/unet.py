import torch

from mri_tools import rA, rfft2,rifft2
from net.unet_parts import *

class Dataconsistency(nn.Module):# In the absence of noise
    def __init__(self):
        super(Dataconsistency, self).__init__()

    def forward(self,x_rec,under_img,sub_mask):
        under_img=under_img.permute(0, 2, 3, 1).contiguous()
        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()
        under_k= rA(x_rec,(1.0 - sub_mask))
        x_k=rfft2(under_img)
        k_out=x_k+under_k
        x_out=rifft2(k_out)
        x_out=x_out.permute(0, 3, 1, 2).contiguous()
        return x_out

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out ,bilinear=True):
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out,x5

class Attention_Ga_Pooling(nn.Module):

    def __init__(self, feature_dim,filters=None):
        super().__init__()
        self.filters=filters
        if self.filters==None:
            self.filters=[feature_dim[1]//2,feature_dim[1]//4,feature_dim[1]//8]
        self.droup_out=nn.Dropout2d(p=0.5)
        self.flow_net = nn.Sequential(
                    nn.Conv2d(feature_dim[1],self.filters[0],kernel_size=(3,3),padding='same'),
                    nn. BatchNorm2d(self.filters[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.filters[0], self.filters[1], kernel_size=(3, 3),padding='same'),
                    nn.BatchNorm2d(self.filters[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.filters[1], self.filters[2], kernel_size=(3, 3),padding='same'),
                    nn.BatchNorm2d(self.filters[2]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.filters[2], 1, kernel_size=(1, 1),padding='valid'),
                    nn.Sigmoid()
                )
        self.conv_att=nn.Conv2d(1,feature_dim[1],kernel_size=(1,1),bias=False,padding='same')
        self.conv_att.weight.data=torch.ones((feature_dim[1],1,1,1))
        self.conv_att.training=False
        self.GAP=nn.AdaptiveAvgPool2d((1,1))
        self.fl=nn.Flatten()
    def forward(self,feature):
        attn_layer=feature#(2,256,16,16)
        attn_layer=self.droup_out(attn_layer)
        attn_layer=self.flow_net(attn_layer)
        attn_layer=self.conv_att(attn_layer)
        mask_attn_layer=torch.multiply(attn_layer,feature)
        gap_features=self.GAP(mask_attn_layer)
        gap_mask=self.GAP(attn_layer)
        gap=(lambda x:x[0]/x[1])([gap_features,gap_mask])
        gap_out=self.fl(gap)
        return gap_out

class Generator(nn.Module):
    def __init__(self,rank):
        super(Generator, self).__init__()
        self.rank=rank
        self.UNet_Ispace = UNet(1,1)
        self.DC=Dataconsistency()
        self.gap=Attention_Ga_Pooling(feature_dim=[2,256,32,2])

    def forward(self,x,under_img,sub_mask):
        out,encoder_out=self.UNet_Ispace(x)
        rec_img = torch.tanh(out + x)
        rec_img=torch.clamp(rec_img,0,1)
        x_dc = self.DC(rec_img, under_img, sub_mask) 
        gap_out=self.gap(encoder_out)
        x_dc = torch.clamp(x_dc, 0, 1)
        return  [x_dc,gap_out]


