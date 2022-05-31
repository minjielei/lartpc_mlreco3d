from cProfile import label
import sys
sys.path.insert(0, '/sdf/home/m/mlei/mlreco3d/uq4dune/torchuq')

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from mlreco.models.experimental.bayes.encoder import MCDropoutEncoder
from torchuq.evaluate.quantile import compute_pinball_loss

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
        self.num_classes = cfg[name].get('num_classes', 1)
        self.final_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.encoder.latent_size, self.num_classes))

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))
        
    def pixel_sum(self, input):
        N = int(input[:, 0].max().item())+1
        res = torch.zeros(N, 1).cuda()
        for i in range(N):
            res[i, 0] = input[input[:, 0]==i][:, 4].sum()
        res.requires_grad_()
        return res

    def forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.final_layer(out)
        res = {
            'pred': [out]
        }
        return res
    
class QuantilePinballLoss(nn.Module):
    def __init__(self, cfg, name='quantile_pinball_loss'):
        super(QuantilePinballLoss, self).__init__()
        self.loss = compute_pinball_loss
    def forward(self, out, gt):
        pred = out['pred'][0]/1000
        labels = gt[0][:, 0]/1000
        n_quantile = pred.shape[1]
        accuracy = torch.abs((pred[:,n_quantile//2]-labels)/labels).mean()
        loss = self.loss(pred, labels)
        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        print(pred[:,n_quantile//2].mean().item() , pred[:,n_quantile//2].std().item() )
        return res

## mean scale error loss
class EnergyMSELoss(nn.Module):
    def __init__(self, cfg, name='energy_mse_loss'):
        super(EnergyMSELoss, self).__init__()
        self.weight = 1
    def forward(self, out, gt):
        pred = out['pred'][0]/1000
        labels = gt[0][:, [0]]/1000
        loss = self.weight * torch.abs((pred-labels)/labels).mean()
        accuracy = torch.abs((pred-labels)/labels).mean()
        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        print(pred.mean().item() , pred.std().item() )
        return res

class EnergyL2Loss(nn.Module):
    def __init__(self, cfg, name='energy_mse_loss'):
        super(EnergyMSELoss, self).__init__()
        self.weight = 1
    def forward(self, out, gt):
        pred = out['pred'][0]/1000
        labels = gt[0][:, [0]]/1000
        loss = self.weight * torch.abs((pred-labels)**2).mean()
        accuracy = torch.abs((pred-labels)/labels).mean()
        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        print(pred.mean().item() , pred.std().item() )
        return res