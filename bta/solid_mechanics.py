# solid_mechanics.py

import numpy as np


def get_mechanical_properties(mat):
    """
    E, mu = solid_mechanics.get_mechanical_properties(mat)
    :param mat:
    :return:
    """
    if mat in ['SILPURAN', 'SILP', 'silpuran']:
        E = 1.2e6
        mu = 0.499
    elif mat in ['ELASTOSIL', 'elastosil']:
        E = 1.2e6
        mu = 0.499
    elif mat in ['PDMS', 'pdms']:
        E = 0.75e6
        mu = 0.499
    elif mat in ['POLYIMIDE', 'POLY', 'polyimide']:
        E = 2.5e9
        mu = 0.34
    else:
        raise ValueError("Only materials ['SILPURAN', 'ELASTOSIL', 'PDMS', and 'POLYIMIDE'] currently available.")
    return E, mu


def flexural_rigidity(E, t, mu):
    return E * t ** 3 / (12 * (1 - mu ** 2))


def circ_center_deflection(p_o, R, D):
    return p_o * R ** 4 / (64 * D)


# Linear Plate Theory: uniformly loaded thin, circular plate
class fSphericalUniformLoad:

    def __init__(self, r, h, youngs_modulus, poisson):
        self.r = r
        self.h = h
        self.E = youngs_modulus
        self.poisson = poisson

    def spherical_uniformly_loaded_clamped_plate_p_e(self, P, E):
        return P * self.r ** 4 / (64 * E * self.h ** 3 / (12 * (1 - self.poisson ** 2)))

    def spherical_uniformly_loaded_simply_supported_plate_p_e(self, P, E):
        return P * self.r ** 2 / (64 * E * self.h ** 3 / (12 * (1 - self.poisson ** 2))) * \
               ((5 + self.poisson) / (1 + self.poisson) * self.r ** 2)


# Bulge Theory: uniformly loaded circular membrane
class fBulgeTheory:

    def __init__(self, r, h, youngs_modulus, poisson):
        self.r = r
        self.h = h
        self.E = youngs_modulus
        self.poisson = poisson

    def linear_elastic_dz_e(self, DZ, E):
        return (1 - 0.24 * self.poisson) * (8 / 3) * (E / (1 - self.poisson)) * (self.h / self.r ** 4) * DZ ** 3

    def linear_elastic_dz_e_sigma(self, DZ, E, SIGMA_0):
        return (1 - 0.24 * self.poisson) * (8 / 3) * (E / (1 - self.poisson)) * (self.h / self.r ** 4) * DZ ** 3 + 4 * (SIGMA_0 * self.h / self.r ** 2) * DZ

    def nonlinear_elastic_dz_e(self, DZ, E):
        return (8 * (1 - 0.24 * self.poisson) * E * self.h * DZ ** 3) / (3 * (1 - self.poisson) * (self.r ** 2 + DZ ** 2) ** 2)

    def nonlinear_elastic_dz_e_sigma(self, DZ, E, SIGMA_0):
        return (8 * (1 - 0.24 * self.poisson) * E * self.h * DZ ** 3) / (3 * (1 - self.poisson) * (self.r ** 2 + DZ ** 2) ** 2) + (4 * SIGMA_0 * self.h * self.r ** 2 * DZ) / ((self.r ** 2 + DZ ** 2) ** 2)