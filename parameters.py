'''
Global parameters, cgs units
'''
import numpy as np
e = 4.8e-10 # elementary charge in statcoulombs
eV_to_erg = 1.60218e-12

############ VALUES TAKEN FROM MESA DAME PLA 2025 TABLE 1 
# CONSTANTS  
e = 4.8e-10 # elementary charge in statcoulombs
eV_to_erg = 1.60218e-12

N_ALPHA = 0.3e13 # cm-3
Z_ALPHA = 2 
CHARGE_ALPHA = 2*e # statC
MASS_ALPHA = 6.64e-24 # g
TEMP_ALPHA = 3.5e6 * eV_to_erg # eV 

N_DEUT = 4.85e13 # cm-3
Z_DEUT = 1
CHARGE_DEUT = e # statC
MASS_DEUT = 3.34e-24 # g
TEMP_DEUT = 15e3 * eV_to_erg # eV 

N_TRIT = 4.85e13 # cm-3
Z_TRIT = 1
CHARGE_TRIT = e # statC
MASS_TRIT = 5.0e-24 # g
TEMP_TRIT = 15e3 * eV_to_erg # eV

N_ELEC = 1.30e14 # cm-3 
CHARGE_ELEC = e # statC
MASS_ELEC = 9.11e-28 # g
Z_ELEC = -1 
TEMP_ELEC = 15e3 * eV_to_erg # eV 

LAMBDA_AD = 24.686 # these values are kind of on the high side, would expect around 15? as a typical value.  
LAMBDA_AT = 24.934 
LAMBDA_AE = 20.355

C_ad = 4 * np.pi * N_DEUT * MASS_ALPHA * Z_ALPHA**2 * Z_DEUT**2 * e**4 * LAMBDA_AD # essentially charge **2 for each species? 
C_at = 4 * np.pi * N_TRIT * MASS_ALPHA * Z_ALPHA**2 * Z_TRIT**2 * e**4 * LAMBDA_AT
C_ae = 4 * np.pi * N_ELEC * MASS_ALPHA * Z_ALPHA**2 * Z_ELEC**2 * e**4 * LAMBDA_AE

V_TH_A = np.sqrt(2 * TEMP_ALPHA/MASS_ALPHA)
V_TH_D = np.sqrt(2 * TEMP_DEUT/MASS_DEUT)
V_TH_T = np.sqrt(2 * TEMP_TRIT/MASS_TRIT)
V_TH_E = np.sqrt(2 * TEMP_ELEC/MASS_ELEC)

TAU_oi = ((C_ad * TEMP_DEUT) / (MASS_DEUT * MASS_ALPHA**2 * V_TH_A**3 * TEMP_ALPHA) \
        + (C_at * TEMP_TRIT) / (MASS_TRIT * MASS_ALPHA**2 * V_TH_A**3 * TEMP_ALPHA) ) ** -1
# print(f'tau_o ions = {TAU_oi}')

ZPAR_NUM = N_DEUT * Z_DEUT**2 * LAMBDA_AD / MASS_DEUT \
         + N_TRIT * Z_TRIT**2 * LAMBDA_AT / MASS_TRIT \
         + N_ELEC * Z_ELEC**2 * LAMBDA_AE / MASS_ELEC
ZPAR_DEN = N_DEUT * Z_DEUT**2 * LAMBDA_AD * TEMP_DEUT / (MASS_DEUT * TEMP_ALPHA) \
         + N_TRIT * Z_TRIT**2 * LAMBDA_AT * TEMP_TRIT / (MASS_TRIT * TEMP_ALPHA) \
         + N_ELEC * Z_ELEC**2 * LAMBDA_AE * TEMP_ELEC / (MASS_ELEC * TEMP_ALPHA) 

ZPAR = ZPAR_NUM / ZPAR_DEN
# print(f'ZPAR = {ZPAR} should match 233.33')

ZPERP_NUM = N_DEUT * Z_DEUT**2 * LAMBDA_AD \
          + N_TRIT * Z_TRIT**2 * LAMBDA_AT \
          + N_ELEC * Z_ELEC**2 * LAMBDA_AE
ZPERP_DEN = N_DEUT * Z_DEUT**2 * LAMBDA_AD * MASS_ALPHA * TEMP_DEUT / (MASS_DEUT * TEMP_ALPHA) \
          + N_TRIT * Z_TRIT**2 * LAMBDA_AT * MASS_ALPHA * TEMP_TRIT / (MASS_TRIT * TEMP_ALPHA) \
          + N_ELEC * Z_ELEC**2 * LAMBDA_AE * MASS_ALPHA * TEMP_ELEC / (MASS_ELEC * TEMP_ALPHA)
ZPERP = 0.5 * ZPERP_NUM/ZPERP_DEN
# print(f'ZPERP = {ZPERP}')

WB_DEN = C_ad * TEMP_DEUT / MASS_DEUT \
       + C_at * TEMP_TRIT / MASS_TRIT \
       + C_ae * TEMP_ELEC / MASS_ELEC 
WPERP_B_DEN = C_ad + C_at + C_ae
WPAR_B_DEN = C_ad / MASS_DEUT \
           + C_at / MASS_TRIT \
           + C_ae / MASS_ELEC 
