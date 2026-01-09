#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: Metric_utils.py                                               #
#   - Description: Transformer-based long-term autoregressive predictor   #
#                                                                         #
#                                               - Created by INHA ACSL    #
#                                                                         #
#  - COPYRIGHT 2025 INHA ACSL. ALL RIGHTS RESERVED.                       #
#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
#   Import Library                                                        #
#-------------------------------------------------------------------------# 

import torch

import numpy as np

# Compute reconstruction MSE
def AE_mse(model, dataset, device, sidx, fidx):
    model.eval()
    
    mses = []
    
    df      = dataset.test_df
    scn_num = dataset.df_scn_test

    _len = fidx - sidx

    model.eval()
    
    with torch.no_grad():
        for idx in np.unique(scn_num):
            data = df[scn_num == idx][sidx:fidx]
            
            # AutoEncoder output
            _in = torch.FloatTensor(data[None,:,:]).to(device)
            out = model(_in).cpu().detach().numpy()[0]

            # Scenario MSE
            mses.append(np.mean((data - out )**2))
        
    return np.mean(mses)

def longtermpredict(idxs, model_auto, model, dataset, config, device, mode='test', save=False):
    n_wdw  = config['n_wdw']
    n_pred = config['n_wdw']
    #----------------------#
    if model_auto is not None:
        model_auto.eval()
    model.eval()
    n_rep  = 13
    t_pred = n_pred * n_rep 

    sidx = 0
    fidx = sidx + t_pred + n_wdw
    
    # Load selected scenario data
    if mode == 'test':
        data = dataset.test_df[np.in1d(dataset.df_scn_test,idxs)].reshape(len(idxs),-1,43)[:,sidx:fidx]
    else:
        data = dataset.train_df[np.in1d(dataset.df_scn_train,idxs)].reshape(len(idxs),-1,43)[:,sidx:fidx]
        
    base = torch.FloatTensor(data[:,:n_wdw,:]).to(device)
    _in  = base[:,:n_wdw].clone().to(device)
    outs = []; out_orgs = []

    # Encode input with AutoEncoder
    if model_auto is not None:
        _in = model_auto.encoder(_in)
        
    tgt         = _in[:,-1:,:].clone().to(device)
    tgt_mask    = model.generate_square_subsequent_mask(n_pred).to(device)        
    
    # Recursive prediction loop
    for j in range(n_rep):
        _ = shortermpredict(_in, tgt, tgt_mask, model, model_auto, config)
        outs.extend(_[0])
        out_orgs.extend(_[1])
        
        _in = torch.cat([*outs[-n_wdw:]], axis=1).to(device)
        tgt = _in[:,-1:,:].clone().to(device)
    
    # Decode latent predictions back to feature space
    if model_auto is not None:
        out_orgs = np.concatenate(out_orgs, axis=1)
    
        return out_orgs, data
    else:
        outs = np.concatenate(outs, axis=1)
        return outs, data

# One-step-ahead predictions repeated n_pred times
def shortermpredict(_in, tgt, tgt_mask, model, model_auto, config):
    n_wdw  = config['n_wdw']
    n_pred = config['n_wdw']
    #-----------------------#

    # Autoregressive one-step predictions
    outs = []; out_orgs = []
    for i in range(n_pred):       
        out = model(_in, tgt, tgt_mask[:i+1,:i+1])
        tgt = torch.cat([tgt, out[:,-1:,:]], axis=1)
        
        if model_auto is not None:
            out_orgs.append(model_auto.decoder(out[:,-1:,:]).cpu().detach())
            
        outs.append( out[:,-1:,:].cpu().detach())
    
    return outs, out_orgs