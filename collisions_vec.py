import numpy as np
import parameters as p
import auxiliary_funcs as af

# PITCH ANGLE ##########################################################
def Asim_xi(v_current, reg=False): # this is already the correction
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    xi = v_current[..., 1]
    drift = -2 * xi / x**3 * (p.ZPERP * af.wperpb_erf(x) - 1 / (4 * x**2) * af.wb_slp(x)) 

    if reg:
        reg_drift = -xi / x**2 * (4 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 2 / (3 * np.sqrt(np.pi)) * af.WB_CUBE)
        return np.where(x < 1e-7, reg_drift, drift)
    return drift


def D_xixi(v_current, reg=True):
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    xi = v_current[..., 1]
    coefficient = 1 / x**3 * (p.ZPERP * af.wperpb_erf(x) - 1 / (4 * x**2) * af.wb_slp(x)) * (1 - xi**2)

    if reg:
        reg_coefficient = 1 / x**2 * (2 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 1 / (3 * np.sqrt(np.pi)) * af.WB_CUBE)
        return np.where(x < 1e-6, reg_coefficient, coefficient)
    return coefficient


# ENERGY ################################################################
def Asim_x(v_current, reg=True): # total energy drift including correction
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    A_a = Aa_x(v_current, reg=reg)

    A_geom = -1 / (2 * x**4) * af.wb_slp(x) + 2 / (x * np.sqrt(np.pi)) * af.wb_cube_exp(x)

    if reg:
        reg_geom = 4 / (3 * np.sqrt(np.pi) * x) * af.WB_CUBE
        return A_a + np.where(x < 1e-7, reg_geom, A_geom)
    return A_a + A_geom


def Aa_x(v_current, reg=True): # original drift vector contribution, without correction
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    drift = -p.ZPAR / x**2 * af.wparb_slp(x)
    if reg:
        reg_drift = -p.ZPAR * 4 / (3 * np.sqrt(np.pi)) * x * af.WPARB_CUBE
        return np.where(x < 1e-10, reg_drift, drift)
    return drift


def D_xx(v_current, reg=True):
    v_current = np.asarray(v_current, dtype=float)
    x = v_current[..., 0]
    coefficient = 1 / (2 * x**3) * af.wb_slp(x)
    if reg:
        return np.where(x < 1e-6, 2 / (3 * np.sqrt(np.pi)) * af.WB_CUBE, coefficient)
    return coefficient


# TOTAL DRIFT VECTOR AND DIFFUSION TENSOR ###############################
# x, xi means full operator 
# nocxn means no correction
# Mesa means Dxx is zero and no correction
# should do Mesa where Dxx is zero

def D_xxi(v_current, reg=True): # full 
    Dxx = D_xx(v_current, reg=reg)
    Dxixi = D_xixi(v_current, reg=reg)
    zeros = np.zeros_like(Dxx)
    return np.stack([
        np.stack([Dxx, zeros], axis=-1),
        np.stack([zeros, Dxixi], axis=-1),
    ], axis=-2)


def A_xxi(v_current, reg=True): # full 
    v_current = np.asarray(v_current, dtype=float)
    A_x = Asim_x(v_current, reg=reg)
    A_xi = Asim_xi(v_current, reg=reg)
    return np.stack([A_x, A_xi], axis=-1)


def D_nopar(v_current, reg=True): # without parallel, so no Dxx 
    Dxixi = D_xixi(v_current, reg=reg)
    zeros = np.zeros_like(Dxixi)
    return np.stack([
        np.stack([zeros, zeros], axis=-1),
        np.stack([zeros, Dxixi], axis=-1),
    ], axis=-2)


def A_nopar(v_current, reg=True): # without parallel diffusion but keeps correction to pitch angle
    v_current = np.asarray(v_current, dtype=float)
    A_x = Aa_x(v_current, reg=reg) # no correction in energy since Dxx=0
    A_xi = Asim_xi(v_current, reg=reg) # correction in pitch angle
    return np.stack([A_x, A_xi], axis=-1)


def A_nocxn(v_current, reg=True): # no correction at all, as used by Mesa Dame
    v_current = np.asarray(v_current, dtype=float)
    A_x = Aa_x(v_current, reg=reg) # no correction in energy 
    A_xi = np.zeros_like(A_x) # no correction in pitch angle, zero
    return np.stack([A_x, A_xi], axis=-1)
