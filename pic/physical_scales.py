import scipy.constants as spc
import math


def debye_length(T_e, n_e):
    """Debye length in meters

    Arguments
    ---------
    T_e : float
        electron temperature in eV
    n_e : float
        electron density in m^{-3}

    Returns
    -------
    lambda_D : float
        Debye length in m
    """
    return math.sqrt((T_e * spc.eV * spc.epsilon_0) / (n_e * spc.elementary_charge**2))


def plasma_ang_frequency(n_e, m=spc.m_p):
    """Plasma angular frequency

    Arguments
    ---------
    n_e : float
        electron density in m^{-3}
    m : float
        ion mass in kg, default to Hydrogen (m_p)

    Returns
    -------
    omega_p : float
        plasma frequency in Hz rad
    """
    return math.sqrt((n_e * spc.elementary_charge**2) / (spc.epsilon_0 * m))



if __name__ == '__main__':
    n_e = 1e4*1e4/(1e-4*100)**3
    T_e = 10
    print("dt for electrons", 1/plasma_ang_frequency(n_e, spc.m_e)*2*spc.pi)
    print("dt for ions", 1/plasma_ang_frequency(n_e, spc.m_p)*2*spc.pi)
    print('Debye length', debye_length(T_e, n_e))
