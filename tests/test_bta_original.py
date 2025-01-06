# tests/test_bta.py

from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.interpolate import splev, splrep, interp1d, UnivariateSpline
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from bta import solid_mechanics
from bta.solid_mechanics import fSphericalUniformLoad


def plot_overlay(tid, dfptid, dfztid, pids, show_plots, save_plots, save_dir):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(dfptid['dt'], dfptid['P'], 'k-o', ms=2, zorder=3.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_xlim([dfztid['dt_rel'].min() - 2.5, dfztid['dt_rel'].max() + 2.5])
    ax1.set_ylabel('Pressue (Pa)')
    ax1.grid(alpha=0.25)

    for pid in pids:
        dfpid = dfztid[dfztid['id'] == pid]
        ax2.plot(dfpid['dt_rel'], dfpid['z'], 'o', color='tab:blue', ms=2, zorder=3, label=pid)
    # ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$z \: (\mu m)$', color='tab:blue')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.075, 1), title=r'$p_{ID}$')
    # ax2.grid(alpha=0.25)
    plt.tight_layout()
    if save_plots:
        plt.savefig(join(save_dir, 'tid{}.png'.format(tid)), dpi=300, facecolor='white')
    elif show_plots:
        plt.show()
    plt.close()

def resample_array(x, y, num_points, sampling_rate=None):
    """
    x, y = process.resample_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    y2 = resample(y, num_points)
    x2 = np.linspace(x.min(), x.max(), num_points)

    return x2, y2

def downsample_array(x, y, num_points, sampling_rate):
    """
    x, y = process.downsample_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    sp1 = splrep(x, y)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = splev(x2, sp1)

    return x2, y2


def interpolate_array(x, y, num_points, sampling_rate=None):
    """
    x, y = process.interpolate_array(x, y, num_points, sampling_rate)
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    f = interp1d(x, y)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = f(x2)

    return x2, y2


def fit_line(x, a, b):
    return a * x + b


if __name__ == '__main__':

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/BulgeTest_200umSILP_4mmDia/analyses'
    SAVE_DIR = join(BASE_DIR, 'fit-w-by-p')

    MEMB_MAT = 'SILPURAN'
    MEMB_RADIUS = 4e-3  # (units: m) 4 mm
    MEMB_THICK = 200e-6  # (units: m) 200 microns

    # FNP = 'combined_P_by_dt.xlsx'
    # FNZ = 'combined_coords_dt-aligned-to-pressure.xlsx'
    FNP = 'fit-w-by-p/P_by_dt__tid17.xlsx'
    FNZ = 'fit-w-by-p/coords_dt-aligned-to-pressure__tid17.xlsx'

    Px, Py = 'dt', 'P'
    Zx, Zy = 'dt_rel', 'z'
    Gx, Gp, Gz = 't', 'P', 'z'

    SHOW_PLOTS = False
    SAVE_PLOTS = False
    SAVE_DF = False

    TIDS = [17]
    PIDS = [0]

    # ---

    tids = TIDS
    pids = PIDS
    save_dir = SAVE_DIR
    base_dir = BASE_DIR
    fnp = FNP
    fnz = FNZ
    show_plots = SHOW_PLOTS
    save_plots = SAVE_PLOTS
    save_df = SAVE_DF

    # -

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dfp_ = pd.read_excel(join(base_dir, fnp))
    dfz_ = pd.read_excel(join(base_dir, fnz))

    if tids is None:
        tids = dfz_['tid'].unique()
    if pids is None:
        pids = dfz_['id'].unique()

    # plot
    dfs = []
    for tid in tids:
        dfptid = dfp_[dfp_['tid'] == tid]
        dfztid = dfz_[dfz_['tid'] == tid]

        # plotting
        plot_P_and_z_by_dt = False
        eval_z_by_P = True

        if plot_P_and_z_by_dt:
            plot_overlay(tid, dfptid, dfztid, pids, show_plots, save_plots, save_dir)

        if eval_z_by_P:
            regularize_pz_on_t = False
            if regularize_pz_on_t:
                # 1. we want to choose one "smooth" particle in order to fit
                # the deflection data onto the pressure data.
                dfztid = dfztid[dfztid['id'] == pids[0]]

                # 2. resample the data onto some physically meaningful grid
                # for example:
                #   Pressure: 5 Pa increments.
                #   Deflection: 1 um increments.

                # 2.1 define grid on x-axis
                # 2.1.a evaluate P and Z wrt x-axis
                # absolute grid extents
                t_min = np.max([dfptid[Px].min(), dfztid[Zx].min()])
                t_max = np.min([dfptid[Px].max(), dfztid[Zx].max()])
                # subset data to grid extents
                sampling_period_P = (dfptid[Px].max() - dfptid[Px].min()) / len(dfptid[Px])
                sampling_period_Z = (dfztid[Zx].max() - dfztid[Zx].min()) / len(dfztid[Zx])
                dfp = dfptid[(dfptid[Px] > t_min - sampling_period_P / 2) &
                             (dfptid[Px] < t_max + sampling_period_P / 2)]
                dfz = dfztid[(dfztid[Zx] > t_min - sampling_period_Z / 2) &
                             (dfztid[Zx] < t_max + sampling_period_Z / 2)]
                # use derivative to define physically meaningful x-grid
                dPdt = (dfp[Py].max() - dfp[Py].min()) / (dfp[Px].max() - dfp[Px].min())
                dZdt = (dfz[Zy].max() - dfz[Zy].min()) / (dfz[Zx].max() - dfz[Zx].min())
                # print out results for evaluation
                print("x-grid extents (t_min, t_max) = ({}, {}) s".format(t_min, t_max))
                print("x-grid span (t_span) = ({}) s".format(np.round(t_max - t_min, 2)))
                print("dt / sample(P) = {} s/#".format(np.round(sampling_period_P, 3)))
                print("dt / sample(Z) = {} s/#".format(np.round(sampling_period_Z, 3)))
                print("sampling rate (P) = {} Hz".format(np.round(1 / sampling_period_P, 3)))
                print("sampling rate (Z) = {} Hz".format(np.round(1 / sampling_period_Z, 3)))
                print("dP/dt = {} Pa/s".format(np.round(dPdt, 2)))
                print("dZ/dt = {} um/s".format(np.round(dZdt, 2)))
                print("dP / sample = {} Pa / #".format(np.round(dPdt * sampling_period_P, 3)))
                print("dZ / sample = {} um / #".format(np.round(dZdt * sampling_period_Z, 3)))
                # ---
                # x-grid of 250 millisecond increments results in ~5 Pa/dt and ~1 um/dt.
                g_dx = 0.25
                gx = np.arange(np.floor(t_min), np.ceil(t_max), g_dx)
                gx = gx[(gx > t_min) & (gx < t_max)]
                # ---
                # 2.1.b resample pressure data
                # since pressure data is inherently smooth,
                # we can simply interpolate onto our grid.
                # convert to numpy array
                xp, yp = dfp[Px].to_numpy(), dfp[Py].to_numpy()
                fp = interp1d(xp, yp)
                gp = fp(gx)
                # -
                # 2.1.c resample deflection data
                # since deflection data is noisy, we fit a
                # 1D spline to smooth the data.
                xz, yz = dfz[Zx].to_numpy(), dfz[Zy].to_numpy()
                spl = UnivariateSpline(xz, yz)
                gz = spl(gx)
                # ---
                # make dataframe with new data on a shared x-grid
                df = pd.DataFrame(np.vstack([gx, gp, gz]).T, columns=[Gx, Gp, Gz])
                df.to_excel(join(save_dir, 'PZ_by_t__tid{}.xlsx'.format(tid)), index=False)
                # -
                # plot to verify
                plot_to_verify = False
                if plot_to_verify:
                    # plot on same subplot
                    fig, ax1 = plt.subplots(figsize=(10, 4))

                    ax1.plot(dfptid[Px], dfptid[Py], 'o', ms=5, color='gray', label='raw pressure')
                    ax1.plot(df[Gx], df[Gp], '-d', ms=2, linewidth=1, color='black', label='fit pressure')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_xlim([df[Gx].min() - 5, df[Gx].max() + 5])
                    ax1.set_ylabel('Pressue (Pa)')
                    ax1.grid(alpha=0.25)
                    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

                    ax2 = ax1.twinx()
                    ax2.plot(dfztid[Zx], dfztid[Zy], 'o', ms=5, color='tab:blue', label='raw z')
                    ax2.plot(df[Gx], df[Gz], '-d', ms=2, linewidth=1, color='red', label='fit z')
                    ax2.set_ylabel(r'$z \: (\mu m)$', color='tab:blue')
                    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.5))

                    plt.tight_layout()
                    #if save_plots:
                    plt.savefig(join(save_dir, 'tid{}_overlay-fit-p-vs-z.png'.format(tid)), dpi=300, facecolor='white')
                    #elif show_plots:
                    # plt.show()
                    plt.close()
                    # -
                    # plot on separate subplots
                    fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=2, sharex=True)

                    ax1.plot(dfptid[Px], dfptid[Py], 'o', ms=5, color='gray', label='raw pressure')
                    ax1.plot(df[Gx], df[Gp], '-d', ms=2, linewidth=1, color='black', label='fit pressure')
                    ax1.set_ylabel('Pressue (Pa)')
                    ax1.grid(alpha=0.25)
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

                    ax2.plot(dfztid[Zx], dfztid[Zy], 'o', ms=5, color='tab:blue', label='raw z')
                    ax2.plot(df[Gx], df[Gz], '-d', ms=2, linewidth=1, color='red', label='fit z')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_xlim([df[Gx].min() - 5, df[Gx].max() + 5])
                    ax2.set_ylabel(r'$z \: (\mu m)$', color='tab:blue')
                    ax2.grid(alpha=0.25)
                    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

                    plt.tight_layout()
                    # if save_plots:
                    plt.savefig(join(save_dir, 'tid{}_overlay-fit-p-and-z.png'.format(tid)), dpi=300, facecolor='white')
                    # elif show_plots:
                    # plt.show()
                    plt.close()

                # ---
            else:
                df = pd.read_excel(join(save_dir, 'PZ_by_t__tid{}.xlsx'.format(tid)))
            # -
            # 1. (optionally) subset data along P or z
            Pmin, Pmax = 0, 250
            # copy original dataframe
            df_ = df.copy()
            # subset data
            df = df[(df[Gp] > Pmin) & (df[Gp] <= Pmax)]
            # -
            # 2. fit function to z vs. P
            P = df[Gp].to_numpy()
            z = df[Gz].to_numpy()
            popt_data, pcov = curve_fit(fit_line, P, z)
            # -
            # 3. resample z vs. P using fit
            fP = np.arange(Pmin, Pmax + 1)
            fz = fit_line(fP, *popt_data)
            # -
            # plot z by P
            """fig, ax = plt.subplots()
            ax.plot(df[Gp], df[Gz], 'k-o', label='data')
            ax.plot(fP, fz, 'r-', label='fit: {}x + {}'.format(np.round(popt[0], 2), np.round(popt[1], 2)))
            ax.set_xlabel('Pressure (Pa)')
            ax.set_ylabel('Deflection (um)')
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()"""

            # ---

            # eval data with respect to solid mechanics
            # -
            # 1. mechanical properties of material
            E, mu = solid_mechanics.get_mechanical_properties(mat=MEMB_MAT)
            # 2. flexural rigidity (based solely on geometry and mechanical properties)
            D = solid_mechanics.flexural_rigidity(E=E, t=MEMB_THICK, mu=mu)
            # 3. analytical deflection
            w = solid_mechanics.circ_center_deflection(p_o=fP, R=MEMB_RADIUS, D=D)
            # convert units
            # w = w * 1e6
            # -
            # fit data to model
            mPlate = fSphericalUniformLoad(r=MEMB_RADIUS, h=MEMB_THICK,
                                           youngs_modulus=E, poisson=mu)
            popt_model, pcov = curve_fit(mPlate.spherical_uniformly_loaded_clamped_plate_p_e, fP, fz * 1e-6)
            mz = mPlate.spherical_uniformly_loaded_clamped_plate_p_e(fP, *popt_model)
            # convert units
            mz = mz * 1e6
            # -

            # ----------------------------------------------------------------------------------------------------------
            # manually input deflection vs. pressure values
            manp = [5, 90, 83, 106, 150, 141, 209, 200, 369, 380, 430, 385, 347, 325, 260, 235, 200, 160, 130, 70, 55, 0, 9]
            manz = [-3, 18, 16, 20, 29, 27, 41, 39, 71, 73, 83, 73, 68, 65, 51, 47, 40, 31, 26, 12, 9, -4, -2]
            manp = np.array(manp)
            manz = np.array(manz)
            # add z-offset to bring z(P = 0) to 0 um.
            manz_offset = 4
            manz = manz + manz_offset
            # ----------------------------------------------------------------------------------------------------------

            # plot z by P
            fig, ax = plt.subplots()
            ax.plot(manp, manz, '^', color='blue', label='Data (tids: 1-14)')
            ax.plot(df[Gp], df[Gz], 'o', color='black', label='Data (tid: 17)')
            # ax.plot(fP, fz, 'k-', label='fit data: {}x + {}'.format(np.round(popt_data[0], 2), np.round(popt_data[1], 2)))
            ax.plot(fP, mz, 'r-', label='Model: ' + r'$\frac {Pr^{4}}{64D} \: (\mu=0.5)$' + '\n'
                                        'Fit: E={} MPa'.format(np.round(popt_model[0] * 1e-6, 2)))
            ax.set_xlabel('Pressure (Pa)')
            ax.set_ylabel('Deflection (um)')
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(save_dir, 'mixed-data_+_model-z-by-P__tid{}.png'.format(tid)), dpi=300, facecolor='white')
            # plt.show()
            plt.close()