import torch.nn as nn
import torch.nn.functional as F

class BYOL_Loss(nn.Module):
    def __init__(self):
        super(BYOL_Loss, self).__init__()

    def forward(self, opred1, opred2, tproj1, tproj2):
        opred1 = F.normalize(opred1, dim=-1, p=2)#Z
        opred2 = F.normalize(opred2, dim=-1, p=2)
        tproj1 = F.normalize(tproj1.detach(), dim=-1, p=2)#q
        tproj2 = F.normalize(tproj2.detach(), dim=-1, p=2)
        loss_part1 = 2 - 2 * (opred1 * tproj2).sum(dim=-1)
        loss_part2 = 2 - 2 * (opred2 * tproj1).sum(dim=-1)
        loss = 0.5 * loss_part1 + 0.5 * loss_part2
        return loss.mean()

