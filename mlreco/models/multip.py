import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from mlreco.models.experimental.bayes.encoder import MCDropoutEncoder

class EnergyRegressor(nn.Module):
    MODULES = ['network_base', 'mcdropout_encoder']
    
    def __init__(self, cfg, name='energy_regressor'):
        super(EnergyRegressor, self).__init__()
        setup_cnn_configuration(self, cfg, 'network_base')
        self.encoder_type = cfg[name].get('encoder_type', 'standard')
        if self.encoder_type == 'standard':
            self.encoder = SparseResidualEncoder(cfg)
        elif self.encoder_type == 'dropout':
            self.encoder = MCDropoutEncoder(cfg)
        else:
            raise ValueError('Unrecognized encoder type: {}'.format(self.encoder_type))
            
        self.final_layer = nn.Linear(self.encoder.latent_size, 1)
        
    def forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.final_layer(out)
        res = {
            'pred': [out]
        }
        return res
    
class EnergyL1Loss(nn.Module):
    def __init__(self, cfg, name='energy_l1_loss'):
        super(EnergyL1Loss, self).__init__()
        self.loss = nn.L1Loss()
    def forward(self, out, gt):
        pred = out['pred'][0]
        labels = gt[0][:, 0]
        loss = self.loss(pred, labels)
        res = {
            'loss': loss
        }
        return res

class EnergyL2Loss(nn.Module):
    def __init__(self, cfg, name='energy_l2_loss'):
        super(EnergyL2Loss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, out, gt):
        pred = out['pred'][0]
        labels = gt[0][:, 0]
        loss = self.loss(pred, labels)
        res = {
            'loss': loss
        }
        return res