
# imports

import numpy as np
from scipy.optimize import minimize  # , curve_fit, fmin_bfgs, minimize_scalar
import functools

"""
from scipy.interpolate import SmoothBivariateSpline
import math

import jax.numpy as jnp
from jax import grad
from jax import random
from jax.config import config
"""


# ---------------------------------------------------- PLANES  ---------------------------------------------------------


def fit_3d_plane(points):
    fun = functools.partial(plane_error, points=points)
    params0 = np.array([0, 0, 0])
    res = minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)

    popt = [a, b, c, d, normal]

    minx = np.min(points[:, 0])
    miny = np.min(points[:, 1])
    maxx = np.max(points[:, 0])
    maxy = np.max(points[:, 1])

    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z, popt


def calculate_z_of_3d_plane(x, y, popt):
    """
    Calculate the z-coordinate of a point lying on a 3D plane.

    :param x:
    :param y:
    :param popt:
    :return:
    """

    a, b, c, d, normal = popt[0], popt[1], popt[2], popt[3], popt[4]

    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]

    return z


def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a * x + b * y + c
    return z


def plane_error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff ** 2
    return result


# ------------------------------------------------ QUASI-2D SURFACES ---------------------------------------------------


def smooth_surface(data, a, b, c):
    """
    3D curved surface function
    """
    x = data[0]
    y = data[1]
    return a * (x ** b) * (y ** c)


# ---------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def calculate_fit_error(fit_results, data_fit_to, fit_func=None, fit_params=None, data_fit_on=None):

    # --- calculate prediction errors
    if fit_results is None:
        fit_results = fit_func(data_fit_on, *fit_params)

    residuals = fit_results - data_fit_to
    r_squared_me = 1 - (np.sum(np.square(residuals))) / (np.sum(np.square(fit_results - np.mean(fit_results))))

    se = np.square(residuals)  # squared errors
    mse = np.mean(se)  # mean squared errors
    rmse = np.sqrt(mse)  # Root Mean Squared Error, RMSE
    r_squared = 1.0 - (np.var(np.abs(residuals)) / np.var(data_fit_to))

    # print("wiki r-squared: {}; old r-squared: {}".format(np.round(r_squared_me, 4), np.round(r_squared, 4)))
    # I think the "wiki r-squared" is probably the correct one...
    # 8/23/22 - the wiki is definitely wrong because values range from +1 to -20...

    return rmse, r_squared