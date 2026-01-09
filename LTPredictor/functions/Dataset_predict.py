#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: Prediction_Dataset.py                                         #
#   - Description: Custom Dataset module for DEC models.                  #
#                                                                         #
#                                               - Created by INHA ACSL    #
#                                                                         #
#  - COPYRIGHT 2025 INHA ACSL. ALL RIGHTS RESERVED.                       #
#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
#   Import Library                                                        #
#-------------------------------------------------------------------------# 

import torch

import pandas as pd
import numpy  as np

from torch.utils.data      import Dataset
from sklearn.preprocessing import MinMaxScaler

from .torch_utils import *

# Slice the sequence by n_wdw and split into X and Y
def crop(data, n_sample, n_wdw, n_pred, n_token):
    X = []; Y = []
    n_data = len(data)
    for _ in range(n_sample):
        sidx    = np.random.randint(0, n_data-n_pred-n_wdw)
        x_idx   = slice(sidx, sidx+n_wdw)
        y_idx   = slice(sidx+n_wdw-n_token, sidx+n_wdw+n_pred)

        X.append(data[x_idx])
        Y.append(data[y_idx])
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

#-------------------------------------------------------------------------#
#   Main                                                                  #
#-------------------------------------------------------------------------#

class custom_dataset(Dataset):
    def __init__(self, n_wdw, n_pred, n_sample, n_token, file_path, selected_class):
        super().__init__()

        #-------------------------------------------------------------------------#
        #   Preprocessing dataset                                                 #
        #-------------------------------------------------------------------------#

        self.selected_class = selected_class
        
        # Extract scenario number and failure class
        df                  = pd.read_csv(file_path)
        scn_num             = df['scn_num'].to_numpy()
        class_              = df['class'].to_numpy()
        selected            = list(df.columns)
        
        removed             = ['Time', 'scn_num', 'class']
        
        for i in removed:
            selected.remove(i)
            
        df                  = df[selected].to_numpy()
        
        # select specific class
        cls_ = np.in1d(class_, selected_class)

        #-------------------------------------------------------------------------#
        #   Spllit train/test dataset                                             #
        #-------------------------------------------------------------------------#
        unique_scn = np.unique(scn_num)          # 실제 존재하는 시나리오 번호
        num_train = int(len(unique_scn) * 0.8)   # 그중 80%
        train_idx = np.random.choice(unique_scn, num_train, replace=False)
        # Randomly sample 80% of scenarios for training
        # train_idx           = np.random.choice(scn_num.max(), int(scn_num.max()*0.8), replace=False) 
        tf                  = np.in1d(scn_num, train_idx)
        train_df            = df[tf]
        train_scn           = scn_num[tf]
        train_cls           = class_[tf]
        
        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(df[tf & cls_])
        
        # Set remaining as test set
        tf                  = ~tf
        test_df             = df[tf]
        test_scn            = scn_num[tf]
        test_cls            = class_[tf]
        
        test_idx            = np.unique(test_scn)
        
        # Apply the scaler to train and test data
        train_df            = scaler.transform(train_df)
        test_df             = scaler.transform(test_df)
        
        self.n_wdw          = n_wdw         # Sliding window size
        self.n_pred         = n_pred        # Prediction horizon
        self.n_token        = n_token       # Overlap length between X and Y
        self.n_sample       = n_sample      # Number of sampled X–Y windows
        
        self.train_df       = train_df
        self.df_scn_train   = train_scn
        
        #-------------------------------------------------------------------------#
        #   Split into X and Y                                                    #
        #-------------------------------------------------------------------------#

        train_df            = self.to_data(train_df, train_scn, train_idx, train_cls)
        X_train, Y_train    = to_tensor(*train_df, func=torch.FloatTensor)       
        
        _                   = self.to_data(test_df, test_scn, test_idx, test_cls)
        X_test, Y_test      = to_tensor(*_, func=torch.FloatTensor)   
        
        #-------------------------------------------------------------------------#
        #   Store processed data                                                  #
        #-------------------------------------------------------------------------# 
                   
        self.df             = df
        self.X_train        = X_train; self.X_test = X_test
        self.Y_train        = Y_train; self.Y_test = Y_test
        
        self.test_df        = test_df
        self.df_scn_test    = test_scn
        self.test_cls       = test_cls
        
        self.n_data         = len(self.X_train)
        
        self.scaler         = scaler
        
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx,:-1], self.Y_train[idx, 1:]
    
    def __len__(self):
        return self.n_data
    
    def to_data(self, df, df_scn, scns, df_cls):
        X = []; Y = []
        for scn in scns:
            tf_scn = df_scn == scn

            # Check the scenario’s class belongs to the selected classes
            if np.unique(df_cls[tf_scn]) not in self.selected_class:
                continue

            data = df[tf_scn]

            # Slice the sequence by n_wdw and split into X and Y
            _X, _Y = crop(data, self.n_sample, self.n_wdw, self.n_pred, self.n_token)
            X.append(_X); Y.append(_Y)
        X = np.vstack(X)
        Y = np.vstack(Y)
        
        return X, Y