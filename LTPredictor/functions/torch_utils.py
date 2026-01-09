#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: torch_utils.py                                                #
#   - Description: Functions for tensor conversion and checkpoint saving. #
#                                                                         #
#                                               - Created by INHA ACSL    #
#                                                                         #
#  - COPYRIGHT 2025 INHA ACSL. ALL RIGHTS RESERVED.                       #
#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
#   Import Library                                                        #
#-------------------------------------------------------------------------# 
import torch

# Convert inputs to tensors using the given constructor
def to_tensor(*x, func):
    out = []
    for _x in x:
        _x = func(_x)
        out.append(_x)
        
    return out

# Save model, optimizer states, and loss history to a checkpoint file
def save(model, optimizer, losses, file_path):

    checkpoint           = {}
    checkpoint['losses'] = losses
    checkpoint['model']  = model.state_dict()
    checkpoint['optim']  = optimizer.state_dict()

    torch.save(checkpoint, file_path)

    return 
