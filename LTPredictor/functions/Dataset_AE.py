#-------------------------------------------------------------------------#
#                                                                         #
#   - Name: AutoEncoder_Dataset.py                                        #
#   - Description: Custom Dataset module for AutoEncoder models.          #
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


# Crop n_sample steps and split into X and Y
def crop(data, data_cls, n_sample):
    n_data      = len(data)

    selected    = np.random.choice(np.arange(n_data), size=n_sample, replace=False)

    X           = np.vstack(data[selected])
    Y           = np.vstack(data_cls[selected])

    return X, Y

#-------------------------------------------------------------------------#
#   Main                                                                  #
#-------------------------------------------------------------------------#

class custom_dataset(Dataset):
    def __init__(self, n_sample, selected_class, file_path):
        super().__init__()

        #-------------------------------------------------------------------------#
        #   Preprocessing dataset                                                 #
        #-------------------------------------------------------------------------#

        self.selected_class     = selected_class
        
        # Extract scenario number and failure class
        df                      = pd.read_csv(file_path)
        scn_num                 = df['scn_num'].to_numpy()
        class_                  = df['class'].to_numpy()
        
        selected                = list(df.columns)
        
        removed                 = ['Time', 'scn_num', 'class']
        
        for i in removed:
            selected.remove(i)

        df                      = df[selected].to_numpy()

        # select specific class
        cls_                    = np.in1d(class_, selected_class)
        
        #-------------------------------------------------------------------------#
        #   Spllit train/test dataset                                             #
        #-------------------------------------------------------------------------#

        # Randomly sample 80% of scenarios for training
        # train_idx               = np.random.choice(scn_num.max(), int(scn_num.max()*0.8) , replace=False)
        # --------------------------------------------------------------------------------------------#
        unique_scn = np.unique(scn_num)          # 실제 존재하는 시나리오 번호
        num_train = int(len(unique_scn) * 0.8)   # 그중 80%
        train_idx = np.random.choice(unique_scn, num_train, replace=False)
        # --------------------------------------------------------------------------------------------#
        
        tf                      = np.in1d(scn_num, train_idx)
        train_df                = df[tf]
        train_scn               = scn_num[tf]
        train_cls               = class_[tf]
        
        # fit scaler
        scaler = MinMaxScaler(); 
        scaler.fit(df[tf & cls_])
        
        # Set remaining as test set
        tf                      = ~tf
        test_df                 = df[tf]
        test_scn                = scn_num[tf]
        test_idx                = np.unique(test_scn)
        test_cls                = class_[tf]
        
        # Apply the scaler to train and test data
        train_df                = scaler.transform(train_df)
        test_df                 = scaler.transform(test_df)
        
        self.n_sample           = n_sample
        
        self.train_df           = train_df
        self.df_scn_train       = train_scn
        
        #-------------------------------------------------------------------------#
        #   Split into X and Y                                                    #
        #-------------------------------------------------------------------------#
        tmp                     = self.to_data(train_df, train_scn, train_idx, train_cls)
        X_train, Y_train        = to_tensor(*tmp, func=torch.FloatTensor)
        
        tmp                     = self.to_data(test_df, test_scn, test_idx, test_cls)
        X_test, Y_test          = to_tensor(*tmp, func=torch.FloatTensor)
        
        #-------------------------------------------------------------------------#
        #   Store processed data                                                  #
        #-------------------------------------------------------------------------#
        
        self.df                 = df

        self.X_train            = X_train
        self.Y_train            = Y_train

        self.X_test             = X_test
        self.Y_test             = Y_test
        
        self.test_df            = test_df
        self.df_scn_test        = test_scn
        
        self.n_data             = len(self.X_train)
        
        self.scaler             = scaler
        
    def __getitem__(self, idx):
        return self.X_train[idx], self.X_train[idx], self.Y_train[idx]
    
    def __len__(self):
        return self.n_data
    
    def to_data(self, df, df_scn, scns, df_class):
        X = []; Y = []
        for scn in scns:
            tf_scn = df_scn == scn

            # Check the scenario’s class belongs to the selected classes
            if np.unique(df_class[tf_scn]) not in self.selected_class:
                continue

            data = df[tf_scn]

            # Crop n_sample steps and split into X and Y
            _X, _Y = crop(data, df_class[tf_scn], self.n_sample)
            X.append(_X)
            Y.append(_Y)
        X = np.vstack(X)
        Y = np.vstack(Y)
        
        return X, Y
    
