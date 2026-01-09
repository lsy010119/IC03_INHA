#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: Train_DEC.py                                                  #
#   - Description: Joint training script for Deep Embedded Clustering     #
#                  (DEC) optimizing both reconstruction and cluster       #
#                  assignment consistency.                                #
#                                                                         #
#                                               - Created by INHA ACSL    #
#                                                                         #
#  - COPYRIGHT 2025 INHA ACSL. ALL RIGHTS RESERVED.                       #
#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
#   Import Library                                                        #
#-------------------------------------------------------------------------# 
import json
import torch
import numpy                    as np
import torch.nn                 as nn
import torch.nn.functional      as F
import matplotlib.pyplot        as plt

from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import StepLR

## custom 
from functions.model_AE         import AutoEncoder
from functions.DEC              import *
from functions.torch_utils      import save
from functions.Dataset_AE       import custom_dataset

def main():
    # --------------------------------------------------------------------- #
    # 1. Configuration & Seed Setup
    # --------------------------------------------------------------------- #
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    # Set seed
    np.random.seed(cfg["seed"])
    device                  = torch.device(cfg["device"])

    cluster_cfg             = cfg["cluster"]
    data_cfg                = cluster_cfg["data"]
    train_cfg               = cluster_cfg["train"]
    
    # --------------------------------------------------------------------- #
    # 2. Dataset Preparation
    # --------------------------------------------------------------------- #
    dataset                 = custom_dataset(
        n_sample            = data_cfg["n_sample"], 
        selected_class      = data_cfg["selected_class"],
        file_path           = data_cfg["file_path"]
    )

    train_dataloader        = DataLoader(
        dataset,
        batch_size          = data_cfg["batch_size"],
        shuffle             = True
    )

    # --------------------------------------------------------------------- #
    # 3. Model Initialization (AutoEncoder for DEC)
    # --------------------------------------------------------------------- #
    num_feat                 = dataset.X_train.shape[-1]

    # Parse hidden dimensions from config
    hidden_dims              = []
    for i, o in cluster_cfg["model"]["hidden_dims"]:
        if i == ":NUM_FEAT":
            i = num_feat
        hidden_dims.append((i, o))

    model_cfg = {
        "dim_hs": hidden_dims,
        "n_cluster": cluster_cfg["model"]["n_cluster"]
    }

    model_auto              = AutoEncoder(**model_cfg).to(device)

    # --------------------------------------------------------------------- #
    # 4. Optimization Setup
    # --------------------------------------------------------------------- #
    optimizer               = torch.optim.Adam(
        model_auto.parameters(),
        lr                  = train_cfg["lr"],
        weight_decay        = train_cfg["weight_decay"]
    )

    scheduler               = StepLR(
        optimizer,
        step_size           = train_cfg["scheduler_step"],
        gamma               = train_cfg["scheduler_gamma"]
    )

    criterion               = nn.MSELoss()
    losses                  = []

    # --------------------------------------------------------------------- #
    # 5. Training Loop (Joint Training)
    # --------------------------------------------------------------------- #
    for i in range(train_cfg["epoch"]):
        batch_loss          = 0
        for _in, tgt, label in train_dataloader:
            optimizer.zero_grad()
            _in             = _in.to(device)
            tgt             = tgt.to(device)

            _output, q      = model_auto(_in, mode='cluster_on')
            p               = target_distribution(q)
            kl_loss         = F.kl_div(q.log(), p, reduction="batchmean")
            
            loss_recon      = criterion(_output, tgt)
            loss            = loss_recon + kl_loss * 0.001

            loss.backward()
            optimizer.step()
            batch_loss += loss_recon.cpu().detach().item()

        losses.append(batch_loss / len(train_dataloader))
        scheduler.step()

        if (i % train_cfg["save_every"]) == 0:
            save(model_auto, optimizer, losses, file_path=train_cfg["save_path"])

    save(model_auto, optimizer, losses, train_cfg["save_path"])

    # --------------------------------------------------------------------- #
    # 6. Plotting Results
    # --------------------------------------------------------------------- #
    plt.figure(figsize=(8,4))
    plt.semilogy(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Autoencoder Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()