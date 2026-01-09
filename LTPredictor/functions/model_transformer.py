#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: Transformer_model.py                                          #
#   - Description: Transformer encoder–decoder for sequence prediction    #
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


'''
- Transformer based Encoder-decoder
- Note:: input embedding == target embedding
'''


class PositionalEncoding(nn.Module):
    def __init__ (self, d_model, dropout, device, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        E_i = torch.arange(0, d_model, 2).reshape(1,-1)
        pos = torch.arange(0, max_len).reshape(-1,1)
        
        # Apply sinusoidal encoding: sin for even, cos for odd indices
        pos_embedding         = torch.zeros((max_len, d_model))
        pos_embedding[:,0::2] = torch.sin( pos / torch.pow(10000, E_i / d_model) )
        pos_embedding[:,1::2] = torch.cos( pos / torch.pow(10000, E_i / d_model) )
        
        self.pos_embedding = pos_embedding.to(device)
        
    def forward (self, x) : 
        _, seq_len, __ = x.size()

        # Add positional encoding to input sequence
        x = x + self.pos_embedding[:seq_len,:]
        
        return self.dropout(x)

class custom_model(nn.Module):
    def __init__ (self, d_model, dropout, nhead, n_layerE, n_layerD, num_feat, n_pred, device):
        super().__init__()
        
        # Positional encoding for encoder/decoder inputs
        self.PE_e = PositionalEncoding(d_model, dropout, device)
        
        # Input embedding network
        self.embedding_e = nn.Sequential(
            nn.Linear(num_feat, d_model//2), 
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        # Transformer encoder–decoder
        self.transformer = nn.Transformer(nhead=nhead, d_model=d_model, 
                                          num_encoder_layers=n_layerE, 
                                          num_decoder_layers=n_layerD, batch_first=True)
        
        # Output projection (d_model → num_feat)
        self.projected = nn.Sequential(
            nn.Linear(d_model, d_model//2), 
            nn.ReLU(),
            nn.Linear(d_model//2, num_feat))
        
        self.num_feat = num_feat
        
    def generate_square_subsequent_mask(self, sz):
        # Autoregressive mask: block access to future positions
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, tgt_mask):
        # Encode source
        src = self.embedding_e(src)
        src = self.PE_e(src)
        
        # Decode target
        tgt = self.embedding_e(tgt)
        tgt = self.PE_e(tgt)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Project back to feature dimension
        out = self.projected(out)
        
        return out
        
