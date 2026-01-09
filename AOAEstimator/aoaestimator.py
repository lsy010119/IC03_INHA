from .base  import *
from .utils import *


class AoAEstimator:
    """
    Kalman Filter & Neural Network Approximation based Angle of Attack Estimator

    Initial Inputs : Initial measuremed states (
        Body Velocity (m/s)             : ndarray(3)
        Wind Velocity (m/s)             : ndarray(3)
        finout (rad)                    : ndarray(4)
        )

    Inputs : Measurements (
        Body Velocity (m/s)             : ndarray(3)
        Specific Acceleration (m/s^2)   : ndarray(3)
        finout (rad)                    : ndarray(4)
        )

    Outputs : Estimated States (
        Body Velocity (m/s)             : ndarray(3)
        Wind Velocity (m/s)             : ndarray(3)
        )
    """

    def __init__(self, init_state : np.ndarray, model_path : str):

        self.Q, self.R, self.P, self.I = get_filter_coeff(device=DEVICE)
        self.aerocoeff_table = load_model(file_path=model_path, device=DEVICE)

        self.x = torch.tensor( init_state.reshape(-1), dtype=torch.float32, device=DEVICE )


    def update(self, vel_body : np.ndarray, acc_specific : np.ndarray, finout : np.ndarray, cI_B : np.ndarray, w_body : np.ndarray):
        vel_body_t      = torch.tensor( vel_body.reshape(1, -1),     dtype=torch.float32, device=DEVICE )
        acc_specific_t  = torch.tensor( acc_specific.reshape(1, -1), dtype=torch.float32, device=DEVICE )
        finout_t        = torch.tensor( finout.reshape(1, -1),       dtype=torch.float32, device=DEVICE )
        cI_B_t          = torch.tensor( cI_B.reshape(1, 3, 3),       dtype=torch.float32, device=DEVICE )
        w_body_t        = torch.tensor( w_body.reshape(1, -1),       dtype=torch.float32, device=DEVICE )

        vbdot = dynamics( vel_b  = self.x[None,0:3],
                          w_b    = w_body_t, 
                          cI_B   = cI_B_t, 
                          a_aero = acc_specific_t )

        Fvb   = Fvb_analytic(w_body_t)[0]

        F = torch.eye( 6, dtype=torch.float32, device=DEVICE )
        F[0:3, 0:3] += Fvb * DT

        xhat_p = torch.zeros_like(self.x)
        xhat_p[0:3] = self.x[0:3] + vbdot.squeeze(0) * DT
        xhat_p[3:6] = self.x[3:6]

        P_pred = F @ self.P @ F.T + self.Q

        xhat_p = xhat_p.clone().detach().requires_grad_(True)
        a_hat, Cxyz_hat = measurement( vel_b  = xhat_p[None, 0:3], 
                                       vel_w  = xhat_p[None, 3:6], 
                                       finout = finout_t, 
                                       model_table = self.aerocoeff_table )

        grad_a = jacob_dydx( y = a_hat, 
                             x = xhat_p[None, :] ).squeeze(0)

        H = torch.zeros( (6,6), device=DEVICE )
        H[0:3,  : ]  = grad_a
        H[3:6, 0:3]  = torch.eye( 3, device=DEVICE )

        y    = torch.cat( [ acc_specific_t, vel_body_t ], dim=1 )
        yhat = torch.cat( [ a_hat, xhat_p[:,0:3] ], dim=1 )

        S = H @ P_pred @ H.T + self.R
        L = torch.linalg.cholesky(S)
        K = torch.cholesky_solve( H @ P_pred.T, L ).T

        xhat_u = xhat_p.T + K @ ( y - yhat ).T
        Phat_u = ( self.I - K @ H ) @ P_pred @ ( self.I - K @ H ).T + K @ self.R @ K.T

        self.x = xhat_u.squeeze()
        self.P = Phat_u.squeeze()


    def get_state(self):
        return self.x.cpu().detach().numpy()