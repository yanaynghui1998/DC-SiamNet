import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one
# from ssdu_masks_create import ssdu_masks
from torch.utils.data import DataLoader
from tqdm import tqdm
from wavelet_transform import IWT,DWT,BBlock#DWT正向小波变换，IWT反向小波变换
# from pytorch_wavelets import DWTForward, DWTInverse
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
class IXIData(Dataset):
    def __init__(self, data_path, u_mask_path,s_mask_up_path,s_mask_down_path, sample_rate):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate

        self.examples = []
        files = list(pathlib.Path(self.data_path).iterdir())
        # The middle slices have more detailed information, so it is more difficult to reconstruct.
        start_id, end_id = 30, 100#30,100
        for file in sorted(files):
            self.examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        if self.sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * self.sample_rate)
            self.examples = self.examples[:num_examples]

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])
        self.mask_net_up = self.mask_under * self.s_mask_up#Omega
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        data = nib.load(file)
        label = data.dataobj[..., slice_id]
        label = normalize_zero_to_one(label, eps=1e-6)
        label = torch.from_numpy(label).unsqueeze(-1).float()
        return label, self.mask_under, self.mask_net_up, self.mask_net_down,file.name, slice_id

# u_mask_path="D:/SSL-MRI-reconstruction/mask_mat/undersampling_mask/mask_2.00x_acs24.mat"
# mask_up_path= "D:/SSL-MRI-reconstruction/mask/selecting_mask/mask_2.00x_acs16.mat"
# mask_down_path= "D:/SSL-MRI-reconstruction/mask/selecting_mask/mask_2.50x_acs16.mat"
# train_loader=IXIData(data_path='D:/SSL-MRI-reconstruction/data/test',u_mask_path=u_mask_path,s_mask_up_path=mask_up_path,s_mask_down_path=mask_down_path, sample_rate=0.02)
# train_loaders= DataLoader(dataset=train_loader, batch_size=1)
# t = tqdm(train_loaders, desc='train' + 'ing', total=int(len(train_loader)))
# #
# for batch in enumerate(t):#batch[0]表示所有矩阵的的总和，int类型，batch[1]是一个列表，有四个元素
#         sample=batch[1][0]#(1,256,256,1)
#         mask_under=batch[1][1]
#         mask_net_up=batch[1][2]
#         mask_net_down=batch[1][3]
# sample=sample.permute(0, 3, 1, 2).contiguous()
# dwt=DWT()
# iwt=IWT()
# # BB=BBlock(in_channels=1,out_channels=1,kernel_size=1,bias=True)
# # sample=BB(sample)
# sample_wavelet=dwt(sample)
# print(sample_wavelet.shape)
# sample_Iwavelet=iwt(sample_wavelet)
# writer=SummaryWriter('MYtensorboard')
# writer.add_images('channl1',sample_wavelet[:,0:1,:,:])
# writer.add_images('channl2',sample_wavelet[:,1:2,:,:])
# writer.add_images('channl3',sample_wavelet[:,2:3,:,:])
# writer.add_images('channl4',sample_wavelet[:,3:4,:,:])
# writer.add_images('recovery',sample_Iwavelet)
# writer.close()