import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.factories import activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.blocks import ResNetBlock
from mlreco.mink.layers.uresnet import UResNetEncoder, UResNetDecoder


class Attention(nn.Module):
    """
    Sparse Attention Module where the feature map is multiplied 
    by a soft masking score tensor (sigmoid activated)
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x, scores):
        features = x.F
        features = features * scores
        coords = x.C
        output = ME.SparseTensor(
            coordinates=coords, feats=features)
        return output


class ExpandAs(nn.Module):
    """
    Given a sparse tensor with one dimensional features, expand the
    feature map to given shape and return a newly constructed 
    ME.SparseTensor.

        - x (ME.SparseTensor): with x.F.shape[1] == 1
        - shape (tuple)
    """
    def __init__(self):
        super(ExpandAs, self).__init__()

    def forward(self, x, shape):
        device = x.F.device
        features = x.F.expand(*shape)
        output = ME.SparseTensor(
            feats=features,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)
        return output


class SPICE(MENetworkBase):

    def __init__(self, cfg, name='spice'):
        super(SPICE, self).__init__(cfg)
        self.model_config = cfg[name]
        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')
        self.embedding_decoder = UResNetDecoder(cfg, name='embedding_decoder')
        self.seed_decoder = UResNetDecoder(cfg, name='seediness_decoder')

        self.num_filters = self.model_config.get('num_filters', 16)
        self.seedDim     = self.model_config.get('seediness_dim', 1)
        self.coordConv   = self.model_config.get('coordConv', False)
        self.sigmaDim    = self.model_config.get('sigma_dim', 1)
        self.seed_freeze = self.model_config.get('seed_freeze', False)
        self.coordConv   = self.model_config.get('coordConv', True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


        self.outputEmbeddings = nn.Sequential(
            ME.MinkowskiBatchNorm(self.num_filters, **self.norm_args),
            ME.MinkowskiLinear(self.num_filters, self.D + self.sigmaDim, 
                               bias=False)
        )

        self.outputSeediness = nn.Sequential(
            ME.MinkowskiBatchNorm(self.num_filters, **self.norm_args),
            ME.MinkowskiLinear(self.num_filters, self.seedDim, bias=False)
        )

        if self.seed_freeze:
            print('Seediness Branch Freezed')
            for p in self.seed_decoder.parameters():
                p.requires_grad = False
            for p in self.outputSeediness.parameters():
                p.requires_grad = False



    def forward(self, input):

        point_cloud, = input
        device = point_cloud.device

        coords = point_cloud[:, 0:self.D+1].to(device)
        features = point_cloud[:, self.D+1:].float().view(-1, 1)

        normalized_coords = (coords[:, 1:self.D+1] - float(self.spatial_size) / 2) \
                          / (float(self.spatial_size) / 2)
        normalized_coords = normalized_coords.float().cuda()
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)


        x = ME.SparseTensor(features, coordinates=coords)

        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        features_cluster = self.embedding_decoder(finalTensor, encoderTensors)
        features_seediness = self.seed_decoder(finalTensor, encoderTensors)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings_feats = embeddings.F
        embeddings_feats[:, :self.D] = self.tanh(embeddings_feats[:, :self.D])
        embeddings_feats[:, :self.D] += normalized_coords
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            'embeddings': [embeddings_feats[:, :self.D]],
            'seediness': [self.sigmoid(seediness.F)],
            'margins': [2 * self.sigmoid(embeddings_feats[:, self.D:])], 
        }
        return res



