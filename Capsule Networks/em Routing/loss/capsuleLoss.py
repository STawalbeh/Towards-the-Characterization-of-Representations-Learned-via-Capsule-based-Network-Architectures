import torch
from CapsuleNetworks.emRouting.loss.spreadLoss import SpreadLoss
from torch.nn.modules.loss import _Loss

class CapsuleLoss(_Loss):
    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super().__init__()
        self.spread = SpreadLoss(m_min=m_min, m_max=m_max, num_class=num_class)
        self.recon = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target, images, reconstruction, r):
        # spread loss
        
        spread_loss = self.spread(x, target, r)
        # scaled reconstruction loss
        
        reconstruction_loss = 5e-4 * self.recon(reconstruction, images.view(*reconstruction.shape))
        loss = spread_loss + reconstruction_loss #5e-4
        
        return loss, 0, 0
