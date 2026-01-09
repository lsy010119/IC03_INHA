#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: DEC.py                                                        #
#   - Description: Soft assignment and target distribution for DEC.       #
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

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim):
        super().__init__()
        self.n_clusters = n_clusters

        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        
        # Xavier initialization for cluster centers
        torch.nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):

        # Student-t distribution-based soft assignment
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

def target_distribution(q):

    # Compute DEC target distribution (p_ij)
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T