#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: AutoEncoder_model.py                                          #
#   - Description: Autoencoder model with an optional clustering head.    #
#                                                                         #
#                                               - Created by INHA ACSL    #
#                                                                         #
#  - COPYRIGHT 2025 INHA ACSL. ALL RIGHTS RESERVED.                       #
#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
#   Import Library                                                        #
#-------------------------------------------------------------------------# 

import torch
import torch.nn as nn

from .DEC import ClusteringLayer


class Decoder(nn.Module):
    def __init__(self, dim_hs):
        super().__init__()
        _list = []

        # Build linear layers for decoding
        for i in dim_hs:
            _list.extend([nn.Linear(*i), nn.ReLU()])

        self.decoder = nn.Sequential(*_list[:-1])

    def forward(self, x):
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dim_hs):
        super().__init__()
        _list = []

        # Build linear layers for encoding
        for i in dim_hs:
            _list.extend([nn.Linear(*i), nn.ReLU()])

        self.decoder = nn.Sequential(*_list[:-1])

    def forward(self, x):
        x = self.decoder(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, dim_hs, n_cluster):
        super().__init__()

        self.encoder = Encoder(dim_hs)
        self.decoder = Decoder([ i[::-1]for i in dim_hs[::-1]])

        # Assigns latent features to clusters via soft probabilities
        if n_cluster != None:
            self.clustering_layer = ClusteringLayer(n_cluster, dim_hs[-1][-1])

        return

    def forward(self, x, mode='cluster_off'):
        latent = self.encoder(x)

        # Reconstructed input
        x_hat  = self.decoder(latent)
        
        # AE-only mode
        if mode == 'cluster_off':
            return x_hat
        
        # Soft cluster assignment
        else:
            q = self.clustering_layer(latent)
            return x_hat, q

