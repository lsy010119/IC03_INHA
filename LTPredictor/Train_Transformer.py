#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: Train_Transformer.py                                          #
#   - Description: Training script for a Transformer-based predictor      #
#                  operating in the latent space of a pre-trained         #
#                  Autoencoder.                                           #
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
import numpy                        as np
import torch.nn                     as nn
import matplotlib.pyplot            as plt

from torch.utils.data               import DataLoader, TensorDataset
from torch.optim.lr_scheduler       import StepLR

# custom
from functions.model_AE             import AutoEncoder
from functions.torch_utils          import save
from functions.metric_utils         import *
from functions.model_transformer    import custom_model
from functions.Dataset_predict      import custom_dataset


def build_ae_model(cfg_ae, num_feat, device):
    hidden_dims = []
    for _in, _out in cfg_ae["hidden_dims"]:
        if _in == ":NUM_FEAT":
            _in = num_feat
        hidden_dims.append((_in, _out))

    ae_input = {
        "dim_hs": hidden_dims,
        "n_cluster": cfg_ae["n_cluster"]
    }

    model_auto = AutoEncoder(**ae_input).to(device)
    checkpoint = torch.load(cfg_ae["checkpoint_path"])
    model_auto.load_state_dict(checkpoint["model"])
    model_auto.eval()

    latent_dim = hidden_dims[-1][1]

    return model_auto, latent_dim


def main():
    # --------------------------------------------------------------------- #
    # 1. Configuration & Seed Setup
    # --------------------------------------------------------------------- #
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    seed                    = cfg["seed"]
    device                  = torch.device(cfg["device"])

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    DEC_cfg                 = cfg["DEC"]
    data_cfg                = DEC_cfg["data"]

    # --------------------------------------------------------------------- #
    # 2. Dataset Preparation
    # --------------------------------------------------------------------- #
    config_dataset = {
        "n_wdw"             : data_cfg["n_wdw"],
        "n_pred"            : data_cfg["n_pred"],
        "n_sample"          : data_cfg["n_sample"],
        "n_token"           : data_cfg["n_token"],
        "selected_class"    : data_cfg["selected_class"]
    }

    file_path               = data_cfg["file_path"]

    dataset                 = custom_dataset(**config_dataset, file_path=file_path)
    train_dataloader        = DataLoader(dataset,
                                  batch_size=data_cfg["batch_size"],
                                  shuffle=True)

    test_dataset            = TensorDataset(dataset.X_test,
                                    dataset.Y_test[:, :-1],
                                    dataset.Y_test[:, 1:])

    test_dataloader         = DataLoader(test_dataset,
                                 batch_size=data_cfg["test_batch_size"])

    # --------------------------------------------------------------------- #
    # 3. Model Building (AE + Transformer)
    # --------------------------------------------------------------------- #

    # AutoEncoder model
    num_feat_raw            = dataset.X_train.shape[-1]

    model_auto, latent_dim  = build_ae_model(DEC_cfg["ae_model"],
                                            num_feat_raw,
                                            device)

    # Transformer model
    trans_cfg               = DEC_cfg["transformer_model"]
    n_pred                  = data_cfg["n_pred"]

    transformer_input = {
        "d_model"           : trans_cfg["d_model"],
        "dropout"           : trans_cfg["dropout"],
        "n_layerE"          : trans_cfg["n_layerE"],
        "n_layerD"          : trans_cfg["n_layerD"],
        "nhead"             : trans_cfg["nhead"],
        "n_pred"            : n_pred
    }

    model                   = custom_model(**transformer_input,
                             num_feat=latent_dim,
                             device=device).to(device)

    # --------------------------------------------------------------------- #
    # 4. Training Setup
    # --------------------------------------------------------------------- #
    criterion               = nn.MSELoss()
    train_cfg               = DEC_cfg["train"]

    optimizer               = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"]
    )

    scheduler               = StepLR(
        optimizer,
        step_size           = train_cfg["scheduler_step"],
        gamma               = train_cfg["scheduler_gamma"]
    )

    epoch                   = train_cfg["epoch"]
    losses                  = []
    
    file_name               = train_cfg["save_path"]

    tgt_len                 = data_cfg["n_wdw"]
    tgt_mask                = model.generate_square_subsequent_mask(tgt_len).to(device)

    # --------------------------------------------------------------------- #
    # 5. Training Loop
    # --------------------------------------------------------------------- #
    model.train()
    # train loop
    for i in range(epoch):
        batch_loss = 0.0

        for (_in, _tgt, _target) in train_dataloader:
            optimizer.zero_grad()

            _in             = model_auto.encoder(_in.to(device))
            _tgt            = model_auto.encoder(_tgt.to(device))
            _target         = model_auto.encoder(_target.to(device))

            # variable window size
            max_len         = dataset.Y_train.shape[1] - 1
            tgt_len         = np.random.randint(1, max_len)
            tgt_mask        = model.generate_square_subsequent_mask(tgt_len).to(device)

            _output         = model(_in,
                            _tgt[:, :tgt_len],
                            tgt_mask[:tgt_len, :tgt_len])

            loss            = criterion(_output, _target[:, :tgt_len])

            loss.backward(); optimizer.step()

            batch_loss += loss.detach().cpu().item()

        losses.append(batch_loss / len(train_dataloader))
        scheduler.step()

        if (i % train_cfg["save_every"]) == 0:
            save(model, optimizer, losses, file_path=file_name)

    save(model, optimizer, losses, file_path=file_name)
    
    # --------------------------------------------------------------------- #
    # 6. Plotting
    # --------------------------------------------------------------------- #
    x = np.arange(len(losses))

    plt.figure(dpi=150, figsize=(10, 8))
    plt.semilogy(x, losses)
    plt.text(x[-1] - max(1, len(x) // 5),
             losses[-1] + 1e-4,
             f"{losses[-1]:.4e}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
