import numpy             as np
import numpy.linalg      as nl
import numpy.random      as nr
import numpy.lib         as nb
import scipy             as sp

import torch
import torch.nn          as nn
import torch.optim       as optim
from   torch.utils.data  import Dataset, DataLoader, TensorDataset

VEL_V       = 1/300
VEL_H       = 1/30
VEL_W       = 1/5
RHO_MEAN    = 1.2
REF_AREA    = 0.01767
DRY_MASS    = 29.7451 
GRAV        = np.array([0,0,9.81])

STD_Q_VEL_BODY = 1e-2
STD_Q_VEL_WIND = 3e-1

STD_M_VEL_BODY = 1e-4

STD_P_VEL_BODY = 1e-2
STD_P_VEL_WIND = 1e-1

DEVICE = torch.device('cpu')
DT = 0.1