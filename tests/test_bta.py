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
from bta.solid_mechanics import fSphericalUniformLoad, fBulgeTheory


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
        ax2.plot(dfpid['dt_rel'], dfpid['z'], 'o', ms=2, zorder=3, label=pid)
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

    BASE_DIR = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250225_C13-20pT-25nmAu_2mmDia/analyses'
    SAVE_DIR = join(BASE_DIR, 'fit-w-by-p')

    MEMB_MAT = 'ELASTOSIL'
    MEMB_RADIUS = 1e-3  # (units: m)
    MEMB_THICK = 20e-6  # (units: m)

    # Use typical values to initialize model
    E, mu = solid_mechanics.get_mechanical_properties(mat=MEMB_MAT)
    # For curve_fit: define guess and lower and upper bounds
    GUESS_E, LB_E, UB_E = 1e6, 1.0e6, 100.0e6  # Estimated by extrapolating Osmani et al. (2016), Fig. 7
    GUESS_SIGMA_0, LB_SIGMA_0, UB_SIGMA_0 = 0.01e6, 0.0, 100.0e6

    FNP = 'combined_P_by_dt.xlsx'
    FNZ = 'combined_coords_dt-aligned-to-pressure.xlsx'

    Px, Py = 'dt', 'P'
    Zx, Zy = 'dt_rel', 'z'
    Gx, Gp, Gz = 't', 'P', 'z'

    dict_pfit = {  # (Fit pressure min, fit pressure max, fit time max)
        1: (10, 1000, 19),
        2: (10, 1000, 17.5),
        3: (10, 1000, 16.75),
        4: (10, 1000, 15.75),
        5: (10, 1000, 17.5),
        6: (10, 1000, 11.8),
        7: (10, 1600, 16.85),
    }

    # ---

    tids = None  # None = all tids
    pids = None  # None = all pids
    save_dir_ = SAVE_DIR
    base_dir = BASE_DIR
    fnp = FNP
    fnz = FNZ
    show_plots = False  # should generally be False
    save_plots = True  # for simplicity, always True
    save_df = True  # for simplicity, always True
    plot_P_and_z_by_dt = True
    eval_z_by_P = True
    regularize_pz_on_t = True
    fit_plate_equation = True
    fit_linear_elastic = True
    fit_nonlinear_elastic = True
    fit_residual_stress = True

    # -

    dfp_ = pd.read_excel(join(base_dir, fnp))
    dfz_ = pd.read_excel(join(base_dir, fnz))

    if tids is None:
        tids = dfz_['tid'].unique()
    if pids is None:
        pids = dfz_['id'].unique()

    # plot
    dfs = []
    results = []
    for tid in tids:
        save_dir = join(save_dir_, 'tid{}'.format(tid))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        dfptid = dfp_[dfp_['tid'] == tid]
        dfztid = dfz_[dfz_['tid'] == tid]

        # plotting

        if plot_P_and_z_by_dt:
            plot_overlay(tid, dfptid, dfztid, pids, show_plots, save_plots, save_dir)

        if eval_z_by_P:
            Pmin, Pmax, Tmax = dict_pfit[tid]

            # co-structure pressure-vs-deflection data
            if regularize_pz_on_t:
                # 1. we want to choose one (or a couple) "smooth" particle(s) in order to fit
                # the deflection data onto the pressure data.
                dfztid = dfztid[dfztid['id'].isin(pids)]
                dfztid = dfztid.groupby(Zx).mean().reset_index()

                # 2. resample the data onto some physically meaningful grid
                dfp = dfptid[(dfptid[Py] >= Pmin) & (dfptid[Py] <= Pmax) & (dfptid[Px] <= Tmax)]

                # 2.1 define grid on x-axis
                # 2.1.a evaluate P and Z wrt x-axis
                # absolute grid extents
                t_min, t_max = dfp[Px].min(), dfp[Px].max()
                # subset data to grid extents
                dfp = dfp[(dfp[Px] >= t_min) & (dfp[Px] <= t_max)]
                dfz = dfztid[(dfztid[Zx] >= t_min) & (dfztid[Zx] <= t_max)]
                # ---
                # x-grid of 250 millisecond increments results in ~5 Pa/dt and ~1 um/dt.
                # or
                # x-grid is deflection x-data
                gx = dfz[Zx].to_numpy()
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
                # spl = UnivariateSpline(xz, yz, s=len(gx) / 25)
                # gz = spl(gx)
                gz = yz
                # ---
                # make dataframe with new data on a shared x-grid
                df = pd.DataFrame(np.vstack([gx, gp, gz]).T, columns=[Gx, Gp, Gz])
                df.to_excel(join(save_dir, 'PZ_by_t__tid{}.xlsx'.format(tid)), index=False)
                # -
                # plot to verify
                plot_to_verify = True
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
                    ax2.plot(dfztid[Zx], dfztid[Zy], 'o', ms=1, color='tab:blue', alpha=0.7, label='raw z')
                    ax2.plot(df[Gx], df[Gz], '-d', ms=2, linewidth=1, color='red', label='fit z')
                    ax2.set_ylabel(r'$z \: (\mu m)$', color='tab:blue')
                    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.5))

                    plt.tight_layout()
                    # if save_plots:
                    plt.savefig(join(save_dir, 'tid{}_overlay-fit-p-vs-z.png'.format(tid)), dpi=300, facecolor='white')
                    # elif show_plots:
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

            # -
            # 1. (optionally) subset data along P or z

            # copy original dataframe
            df_ = df.copy()
            # subset data
            df = df[(df[Gp] > Pmin) & (df[Gp] <= Pmax) & (df[Gx] <= Tmax)]

            # ---

            if fit_plate_equation:

                # 2. fit function to z vs. P
                P = df[Gp].to_numpy()
                z = df[Gz].to_numpy()
                popt_data, pcov = curve_fit(fit_line, P, z)
                fit_line_slope, fit_line_y_intercept = np.round(popt_data[0], 4), np.round(popt_data[1], 3)

                # -
                # 3. resample z vs. P using fit
                fP = np.arange(np.min([0, Pmin]), P.max() + 5)
                fz = fit_line(fP, *popt_data)
                # -
                # plot z by P
                plot_fit_line = True
                if plot_fit_line:
                    fig, ax = plt.subplots()
                    ax.plot(df[Gp], df[Gz], 'k-o', label='data')
                    ax.plot(fP, fz, 'r-', label='fit: {}x + {}'.format(np.round(popt_data[0], 4), np.round(popt_data[1], 2)))
                    ax.set_xlabel('Pressure (Pa)')
                    ax.set_ylabel('Deflection (um)')
                    ax.grid(alpha=0.25)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(join(save_dir, 'old__tid{}_overlay-fit-p-and-z.png'.format(tid)), dpi=300, facecolor='white')
                    # ---
                # ---
                # eval data with respect to solid mechanics
                # -
                # flexural rigidity (based solely on geometry and mechanical properties; i.e., not on pressure)
                D = solid_mechanics.flexural_rigidity(E=E, t=MEMB_THICK, mu=mu)
                # analytical deflection
                w = solid_mechanics.circ_center_deflection(p_o=fP, R=MEMB_RADIUS, D=D)

                # -
                # fit data to model
                mPlate = fSphericalUniformLoad(r=MEMB_RADIUS, h=MEMB_THICK, youngs_modulus=E, poisson=mu)
                popt_model, pcov = curve_fit(mPlate.spherical_uniformly_loaded_clamped_plate_p_e, fP, fz * 1e-6)
                plate_model_E = np.round(popt_model[0] * 1e-6, 2)
                mz = mPlate.spherical_uniformly_loaded_clamped_plate_p_e(fP, *popt_model)
                # convert units
                mz = mz * 1e6
                # -
                # plot z by P
                plot_fit_model = True
                if plot_fit_model:
                    fig, ax = plt.subplots()
                    ax.plot(df[Gp], df[Gz], 'o', color='black', label='Data (tid: 17)')
                    ax.plot(fP, mz, 'r-',
                            label='Model: ' + r'$\frac {Pr^{4}}{64D} \: (\mu=$' + str(mu) + r'$)$' +
                                  '\n' + 'Fit: E={} MPa'.format(np.round(popt_model[0] * 1e-6, 2)))
                    ax.set_xlabel('Pressure (Pa)')
                    ax.set_ylabel('Deflection (um)')
                    ax.grid(alpha=0.25)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(join(save_dir, 'mixed-data_+_plate-model-z-by-P__tid{}.png'.format(tid)), dpi=300, facecolor='white')
                    plt.close()

            # ---

            if fit_linear_elastic:
                # fit function to z vs. P
                P = df[Gp].to_numpy()
                z = df[Gz].to_numpy()
                # extrapolate z-space
                fz = np.linspace(0, np.max(z), 25)
                # ---
                # set up model
                mBulgeLinear = fBulgeTheory(r=MEMB_RADIUS, h=MEMB_THICK, youngs_modulus=E, poisson=mu)
                # -
                # fit model
                if fit_residual_stress:
                    popt_model, pcov = curve_fit(f=mBulgeLinear.linear_elastic_dz_e_sigma,
                                                 xdata=z * 1e-6,
                                                 ydata=P,
                                                 p0=[GUESS_E, GUESS_SIGMA_0],
                                                 bounds=([LB_E, LB_SIGMA_0], [UB_E, UB_SIGMA_0]),
                                                 )
                    linear_model_E, linear_model_sigma_0 = np.round(popt_model[0] * 1e-6, 2), np.round(popt_model[1] * 1e-3, 1)
                    mP = mBulgeLinear.linear_elastic_dz_e_sigma(fz * 1e-6, *popt_model)
                    lbl = 'Fit: E={} MPa, sigma_0={} kPa'.format(np.round(popt_model[0] * 1e-6, 2),
                                                                 np.round(popt_model[1] * 1e-3, 1))
                else:
                    popt_model, pcov = curve_fit(f=mBulgeLinear.linear_elastic_dz_e,
                                                 xdata=z * 1e-6,
                                                 ydata=P, p0=[GUESS_E],
                                                 bounds=(LB_E, UB_E),
                                                 )
                    mP = mBulgeLinear.linear_elastic_dz_e(fz * 1e-6, *popt_model)
                    lbl = 'Fit: E={} MPa'.format(np.round(popt_model[0] * 1e-6, 2))
                # -
                # ---
                # plot z by P
                fig, ax = plt.subplots()
                ax.plot(df[Gz], df[Gp], 'o', color='black', label='Data (tid: {})'.format(tid))
                ax.plot(fz, mP, 'r-', label=lbl)
                ax.set_xlabel('Deflection (um)')
                ax.set_ylabel('Pressure (Pa)')
                ax.grid(alpha=0.25)
                ax.legend()
                plt.tight_layout()
                plt.savefig(join(save_dir, 'mixed-data_+_linear-elastic-model-z-by-P__tid{}.png'.format(tid)),
                            dpi=300, facecolor='white')
                plt.close()

            # ---

            if fit_nonlinear_elastic:
                # fit function to z vs. P
                P = df[Gp].to_numpy()
                z = df[Gz].to_numpy()
                # extrapolate z-space
                fz = np.linspace(0, np.max(z), 25)
                # ---
                # set up model
                mBulgeNonLinear = fBulgeTheory(r=MEMB_RADIUS, h=MEMB_THICK, youngs_modulus=E, poisson=mu)
                # fit data to model
                if fit_residual_stress:
                    popt_model, pcov = curve_fit(f=mBulgeNonLinear.nonlinear_elastic_dz_e_sigma,
                                                 xdata=z * 1e-6,
                                                 ydata=P,
                                                 p0=[GUESS_E, GUESS_SIGMA_0],
                                                 bounds=([LB_E, LB_SIGMA_0], [UB_E, UB_SIGMA_0]),
                                                 )
                    nonlinear_model_E, nonlinear_model_sigma_0 = np.round(popt_model[0] * 1e-6, 2), np.round(popt_model[1] * 1e-3, 1)
                    mP = mBulgeNonLinear.nonlinear_elastic_dz_e_sigma(fz * 1e-6, *popt_model)
                    lbl = label='Fit: E={} MPa, sigma_0={} kPa'.format(np.round(popt_model[0] * 1e-6, 2),
                                                                     np.round(popt_model[1] * 1e-3, 1))
                else:
                    popt_model, pcov = curve_fit(f=mBulgeNonLinear.nonlinear_elastic_dz_e,
                                                 xdata=z * 1e-6,
                                                 ydata=P,
                                                 p0=GUESS_E,
                                                 bounds=(LB_E, UB_E),
                                                 )
                    mP = mBulgeNonLinear.nonlinear_elastic_dz_e(fz * 1e-6, *popt_model)
                    lbl = label = 'Fit: E={} MPa'.format(np.round(popt_model[0] * 1e-6, 2))
                # ---
                # plot z by P
                fig, ax = plt.subplots()
                ax.plot(df[Gz], df[Gp], 'o', color='black', label='Data (tid: {})'.format(tid))
                ax.plot(fz, mP, 'r-', label=lbl)
                ax.set_xlabel('Deflection (um)')
                ax.set_ylabel('Pressure (Pa)')
                ax.grid(alpha=0.25)
                ax.legend()
                plt.tight_layout()
                plt.savefig(join(save_dir, 'mixed-data_+_nonlinear-elastic-model-z-by-P__tid{}.png'.format(tid)),
                            dpi=300, facecolor='white')
                plt.close()

            # ---

            # append data
            res = [tid, MEMB_RADIUS * 1e3, MEMB_THICK * 1e6, df[Gp].min(), df[Gp].max(),
                   fit_line_slope, fit_line_y_intercept,
                   D, plate_model_E,
                   linear_model_E, linear_model_sigma_0 * 1e-3,
                   nonlinear_model_E, nonlinear_model_sigma_0 * 1e-3,
                   GUESS_E * 1e-6, LB_E * 1e-6, UB_E * 1e-6,
                   GUESS_SIGMA_0 * 1e-6, LB_SIGMA_0 * 1e-6, UB_SIGMA_0 * 1e-6]
            results.append(res)

    df_res = pd.DataFrame(
        np.array(results),
        columns=[
            'tid', 'radius_mm', 'thickness_um', 'Pmin_Pa', 'Pmax_Pa',
            'fit_line_slope_um_per_Pa', 'fit_line_y_intercept_um',
            'flexural_rigidity', 'plate_model_E_MPa',
            'linear_model_E_MPa', 'linear_model_sigma_0_MPa',
            'nonlinear_model_E_MPa', 'nonlinear_model_sigma_0_MPa',
            'GUESS_E_MPa', 'LB_E_MPa', 'UB_E_MPa',
            'GUESS_SIGMA_0_MPa', 'LB_SIGMA_0_MPa', 'UB_SIGMA_0_MPa',
        ],
    )
    df_res.to_excel(join(save_dir_, 'results-overview.xlsx'))

# ---

print("Script completed without errors.")