from net.unet import *
from image_contrastive_loss import BYOL_Loss
class SiameseNet(nn.Module):
    def __init__(self,rank,num_layers):
        super(ISTANetPlus, self).__init__()
        self.rank=rank
        self.num_layers = num_layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(Generator(self.rank))
        self.layers = nn.ModuleList(self.layers)
    def forward(self,inp,sub_mask):
        x = inp
        encoder = []
        for i in range(self.num_layers):
            [x,encoder_out]= self.layers[i](x,inp,sub_mask)
            encoder.append(encoder_out)
        x_final = x
        return [x_final,encoder]

class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128,out_dim=256): 
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,out_dim)
                )
    def forward(self, x):
        x = self.layer(x)
        return x

class ParallelNetwork(nn.Module):
    def __init__(self,rank,num_layers):
        super(ParallelNetwork, self).__init__()
        self.rank=rank
        self.num_layers = num_layers
        self.network = SiameseNet(self.rank,self.num_layers)
        self.project_head=MLP()
        self.predict_head=MLP()
        self.conloss=BYOL_Loss()
    def forward(self, under_img_up,mask_up,under_img_down,mask_down):
        output_up,encoder_up = self.network(under_img_up,mask_up)
        output_down,encoder_down = self.network(under_img_down,mask_down)
        encoder_up_cate = torch.cat(encoder_up, dim=0)
        encoder_down_cate = torch.cat(encoder_down, dim=0)
        encoder_up_tensor=self.project_head(encoder_up_cate)
        target_up_tensor = self.predict_head(encoder_up_tensor)
        encoder_down_tensor = self.project_head(encoder_down_cate)
        target_down_tensor=self.predict_head(encoder_down_cate)
        loss=self.conloss(target_up_tensor,target_down_tensor,encoder_up_tensor,encoder_down_tensor)
        return output_up,output_down,loss
