import numpy as np
import parameters as p
import auxiliary_funcs as af

# Karney Perpendicular Diffusion 
# worry about regularizing later 

def A_karney(v_current, omega, E_0, k_perp): 
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    xi = v_current[..., 1]

    resonance = k_perp**2 * p.V_TH_A**2 * x**2 * (1 - xi**2) - omega**2
    mask = resonance > 0
    A = np.zeros_like(v_current, dtype=float)

    F = 0.5 * ((p.CHARGE_ALPHA * E_0 * omega)/(p.c * p.MASS_ALPHA * k_perp * p.V_TH_A * p.V_TH_A))**2

    resonant = resonance[mask] # pick out where it's true
    x_res, xi_res = x[mask], xi[mask]
    A_x = - k_perp**2 * p.V_TH_A**2 * x_res / resonant**(3/2) - 1 / x_res / resonant**(1/2)
    A_xi = k_perp**2*p.V_TH_A**2 * xi_res / resonant**(3/2) + 3*xi_res / x_res**2 / resonant**(1/2)
    A[mask] = np.stack([A_x, A_xi], axis=-1)

    return A


def D_karney(v_current, omega, E_0, k_perp):
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    xi = v_current[..., 1]

    resonance = k_perp**2 * p.V_TH_A**2 * x**2 * (1 - xi**2) - omega**2
    mask = resonance > 0
    D = np.zeros((*x.shape,2,2))

    F = 0.5 * ((p.CHARGE_ALPHA * E_0 * omega)/(p.c * p.MASS_ALPHA * k_perp * p.V_TH_A * p.V_TH_A))**2
    resonant = resonance[mask] # pick out where it's true
    x_res, xi_res = x[mask], xi[mask]

    Dxx = 1 / x_res**2 * resonant**(1/2)
    Dcross = - xi_res / x_res**3 * resonant**(1/2)
    Dxixi = xi_res**2 / x_res**4 * resonant**(1/2)
    D[mask] = np.stack([
                    np.stack([Dxx, Dcross], axis=-1),
                    np.stack([Dcross, Dxixi], axis=-1),
                ], axis=-2) 

    return D