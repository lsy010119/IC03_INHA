from .base import *


class AeroCoeff(nn.Module):
    """
    Aero Coefficient Estimator Model
    MLP structure to estimate aerodynamic coefficients
    
    Inputs  : 10 features (
                Body velocity (scaled) : ndarray(3)
                Wind velocity (scaled) : ndarray(3)
                finout (rad)           : ndarray(4)
                )
    Outputs : 3 coefficients (Cx, Cy, Cz)
    """

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(10,  32), nn.GELU(), 
            nn.Linear(32, 3))

    def forward(self, x):
        cf  = self.mlp(x)
        return cf
    