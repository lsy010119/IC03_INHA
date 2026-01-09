from .base import *
from .models import AeroCoeff


def data_scaling( X_trn : np.ndarray, Y_trn : np.ndarray ):

    X_scale_param = np.zeros((X_trn.shape[1],2))
    Y_scale_param = np.zeros((Y_trn.shape[1],2))

    for i in range( X_trn.shape[1] ):
        X_scale_param[i,0] = np.mean( X_trn[:,i] )
        X_scale_param[i,1] = np.std ( X_trn[:,i] )

    for i in range( Y_trn.shape[1] ):
        Y_scale_param[i,0] = np.mean( Y_trn[:,i] )
        Y_scale_param[i,1] = np.std ( Y_trn[:,i] )

    return X_scale_param, Y_scale_param


def dynamics(vel_b : torch.Tensor, w_b : torch.Tensor, cI_B : torch.Tensor, a_aero : torch.Tensor):
    N           = vel_b.shape[0]
    crossw_bvel_b = torch.cross( w_b, vel_b, dim=1 )
    ag = torch.tensor( GRAV, dtype=torch.float32 ).to( vel_b.device )

    R_i_b = cI_B.view( N, 3, 3 )
    R_ag  = torch.matmul( R_i_b, ag.view( 1, 3, 1 ) ).squeeze(-1)
    
    vbdot = - crossw_bvel_b + a_aero + R_ag

    return vbdot
    

def Fvb_analytic(omega_b : torch.Tensor):
    wx = omega_b[:, 0]
    wy = omega_b[:, 1]
    wz = omega_b[:, 2]

    zeros = torch.zeros_like(wx)

    row1 = torch.stack([zeros,  wz,   -wy], dim=-1)
    row2 = torch.stack([-wz,   zeros,  wx], dim=-1)
    row3 = torch.stack([wy,   -wx,   zeros], dim=-1)

    Fvb = torch.stack([row1, row2, row3], dim=-2)
    return Fvb


def measurement(vel_b : torch.Tensor, vel_w : torch.Tensor, finout : torch.Tensor, model_table : nn.Module):
    _in_table = torch.cat( [vel_b[:,0:1] * VEL_V, 
                            vel_b[:,1:]  * VEL_H, 
                            vel_w        * VEL_W, 
                            finout ], dim=1 )
    
    _Cxyzhat  = model_table(_in_table)

    q      = 0.5 * RHO_MEAN * torch.sum( ( vel_b - vel_w )**2, dim=1, keepdim=True )
    a_aero = q * REF_AREA * _Cxyzhat / DRY_MASS

    return a_aero, _Cxyzhat


def jacob_dydx(y : torch.Tensor, x : torch.Tensor):
    b, ny = y.size()
    _, nx = x.size()
    
    jacob_dydx = torch.zeros(b, ny, nx, device=y.device)
    
    for i in range(ny):
        grad_x = torch.autograd.grad(outputs        = y[:, i].sum(), 
                                     inputs         = x,
                                     create_graph   = False,
                                     retain_graph   = (i < ny-1) )[0]
        
        jacob_dydx[:, i, :] = grad_x
    
    return jacob_dydx


def get_filter_coeff(device: torch.device):
    Q = torch.diag( torch.tensor( [STD_Q_VEL_BODY**2, STD_Q_VEL_BODY**2, STD_Q_VEL_BODY**2,
                                   STD_Q_VEL_WIND**2, STD_Q_VEL_WIND**2, STD_Q_VEL_WIND**2], dtype=torch.float32 ) ).to(device)
    
    R = torch.diag( torch.tensor( [0.05**2,          1**2,               1**2,
                                   STD_M_VEL_BODY**2, STD_M_VEL_BODY**2, STD_M_VEL_BODY**2], dtype=torch.float32 ) ).to(device)

    P = torch.diag( torch.tensor( [STD_P_VEL_BODY**2, STD_P_VEL_BODY**2, STD_P_VEL_BODY**2,
                                   STD_P_VEL_WIND**2, STD_P_VEL_WIND**2, STD_P_VEL_WIND**2], dtype=torch.float32 ) ).to(device)
    
    I = torch.eye(6, dtype=torch.float32).to(device)
    
    return Q, R, P, I


def load_model(file_path: str, device: torch.device):
    model = AeroCoeff().to( device )
    checkpoint = torch.load( file_path, map_location=device )
    model.load_state_dict( checkpoint['model'] )
    model.eval()

    return model