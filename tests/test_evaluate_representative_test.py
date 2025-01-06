

import os
from os.path import join

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline

from utils import fit, io

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

base_dir = '/Users/mackenzie/Desktop/Bulge Test/Experiments/20250105_C9-0pT_NoMetal_3mmDia'
read_dir = join(base_dir, 'results/coords')
filename = 'test_coords_test3'

path_results = join(base_dir, 'representative_test')
path_results_pids = join(path_results, 'pids')
path_results_pre_start = join(path_results, 'pre-start')
path_results_bispl = join(path_results, 'bispl')

for pth in [path_results, path_results_pids, path_results_pre_start, path_results_bispl]:
    if not os.path.exists(pth):
        os.makedirs(pth)

# 0. experimental parameters
scale_z = 1
frame_rate = 11.001
padding = 25
num_pixels = 512
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding

# 1. read
df = io.read_coords(filepath=join(read_dir, filename + '.xlsx'),
                    scale_z=scale_z,
                    z0=0,
                    flip_z=True,
                    frame_rate=frame_rate,
                    skip_frame_zero=True,
                    only_pids=None)

# 2. PROCESS: plot pids to determine "good" pids
start_frame = 8  # average z(frame < start_frame) to estimate dz = 0
end_frames = (215, 235)  # average z(frame > end_frame) to estimate dz_max
plot_pids = True
if plot_pids:
    # fitting parameters
    sx = 0.5  # 1: smoothing factor (sx * number of samples to fit)
    pdeg = 12  # 12: degree of polynominal

    px, py = 'frame', 'z'
    pids = df['id'].unique()
    pids.sort()

    y_means = []
    for pid in pids:
        dfpid = df[df['id'] == pid]

        # dz analysis
        y_mean_i = dfpid[dfpid[px] < start_frame][py].mean()
        y_mean_f = dfpid[(dfpid[px] > end_frames[0]) & (dfpid[px] < end_frames[1])][py].mean()
        dy_mean = np.round(y_mean_f - y_mean_i, 2)

        fig, (ax, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # plot "raw" 3D particle tracking coords
        ax.plot(dfpid[px], dfpid[py], 'o', ms=1, label=pid, alpha=0.8, zorder=3.5)

        # fit on subset of 3D coords
        x = dfpid[dfpid[px] < end_frames[1]][px].to_numpy()
        y = dfpid[dfpid[px] < end_frames[1]][py].to_numpy()

        # fit spline
        s = len(x) * sx
        tck_s = splrep(x, y, s=s)
        sq_errors_spl = (y - BSpline(*tck_s)(x)) ** 2
        rmse_spl = np.round(np.sqrt(np.mean(sq_errors_spl)), 3)

        # fit polynomial
        pf = np.polyfit(x, y, deg=pdeg)
        p12 = np.poly1d(pf)
        sq_errors_pf = (y - np.poly1d(pf)(x)) ** 2
        rmse_pf = np.round(np.sqrt(np.mean(sq_errors_pf)), 3)

        p1, = ax.plot(x, BSpline(*tck_s)(x), '-', linewidth=0.5, alpha=0.8, label='sx={}'.format(sx))
        ax2.plot(x, sq_errors_spl, '-o', color=p1.get_color(), linewidth=0.5, ms=1, alpha=1,
                 label='rmse={}'.format(rmse_spl))

        p2, = ax.plot(x, np.poly1d(pf)(x), '-', linewidth=0.5, alpha=0.8, label='deg={}'.format(pdeg))
        ax2.plot(x, sq_errors_pf, '-o', color=p2.get_color(), linewidth=0.5, ms=1, alpha=1,
                 label='rmse={}'.format(rmse_pf))

        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_title(r'$\Delta z=$' + ' {} '.format(dy_mean) + r'$\mu m$')
        ax2.legend()
        ax2.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(join(path_results_pids, 'pid{}.png'.format(pid)), dpi=300, facecolor='w')
        plt.close()

        y_means.append([pid, start_frame, end_frames[0], end_frames[1], y_mean_i, y_mean_f, dy_mean, sx, pdeg, rmse_spl, rmse_pf])

    df_y_means = pd.DataFrame(np.array(y_means), columns=['pid', 'start_frame', 'end_frame_i', 'end_frame_f',
                                                          'y_mean_i', 'y_mean_f', 'dy_mean',
                                                          'sx', 'pdeg', 'rmse_spl', 'rmse_pf'])
    df_y_means = df_y_means.sort_values(by='dy_mean', ascending=False)
    df_y_means.to_excel(join(path_results_pids, 'dy-mean_per_pid.xlsx'), index=False)

    raise ValueError("Pause here to evaluate the results and pick out 'best' and 'bad' pids.")

# 2. RESULTS
good_pids = np.arange(0, 28, 1)
best_pids = [5]  # z-tracking is excellent (ideally, located at r = 0 (i.e., max deflection)
ok_pids = []  # z-tracking may be useful for analysis
bad_pids = []  # data will be thrown out b/c not useful

# not "bad" pids
df = df[~df['id'].isin(bad_pids)]
# or, only "best" pids
df = df[df['id'].isin(good_pids)]

# ---

# 3. PROCESS: fit plane to "good" pids before deflection
pre_deflection_analysis = False
if pre_deflection_analysis:
    px, py = 'frame', 'z'
    frames = df['frame'].unique()

    pre_starts = [55]
    for pre_start in pre_starts:

        res = []
        for fr in frames:
            dfr = df[df['frame'] == fr]

            if fr < np.min([start_frame * pre_start, end_frames[0]]):
                # fit plane
                zmean = dfr[py].mean()
                zstd = dfr[py].std()

                # fit plane (x, y, z units: pixels)
                points_pixels = np.stack((dfr.x, dfr.y, dfr[py])).T
                px_pixels, py_pixels, pz_pixels, popt_pixels = fit.fit_3d_plane(points_pixels)

                # calculate fit error
                fit_results = fit.calculate_z_of_3d_plane(dfr.x, dfr.y, popt=popt_pixels)
                rmse_plane, r_squared_plane = fit.calculate_fit_error(fit_results, data_fit_to=dfr[py].to_numpy())

                # calculate zf at image center: (x = 256 + padding, y = 256 + padding)
                zm_xyc = fit.calculate_z_of_3d_plane(img_xc, img_yc, popt=popt_pixels)

                res_ = [fr, zmean, zstd, zm_xyc, rmse_plane, r_squared_plane]
                res.append(res_)

        dfr = pd.DataFrame(np.array(res), columns=['frame', 'zmean', 'zstd', 'zxyc', 'rmse', 'r_squared'])
        dfr.to_excel(join(path_results_pre_start, 'pre-start-{}X.xlsx'.format(pre_start)), index=False)

        fig, (ax, ax2) = plt.subplots(figsize=(9, 5), nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax.errorbar(dfr['frame'], dfr['zmean'], yerr=dfr['zstd'], fmt='-o', ms=1, capsize=1, elinewidth=0.5, label='raw')
        ax.errorbar(dfr['frame'], dfr['zxyc'], yerr=dfr['rmse'], fmt='-o', ms=1, capsize=1, elinewidth=0.5, label='fit plane')
        ax.set_ylabel('z')
        ax.grid(alpha=0.25)
        ax.set_title('mean(z)(raw, plane) = ({}, {})'.format(np.round(dfr['zmean'].mean(), 2), np.round(dfr['zxyc'].mean(), 2)))
        ax.legend()

        ax2.plot(dfr['frame'], dfr['zstd'], '-o', ms=1, label='raw')
        ax2.plot(dfr['frame'], dfr['rmse'], '-o', ms=1, label='fit plane')
        ax2.set_ylabel('r.m.s. error (z)')
        ax2.set_xlabel('frame')
        ax2.set_xticks(np.arange(0, dfr['frame'].max() + 5, 5))
        ax2.set_yticks(np.arange(0, dfr['zstd'].max() + 0.2, 0.5))
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(join(path_results_pre_start, 'pre-start-{}X.png'.format(pre_start)))
        plt.close()

    raise ValueError("Pause to evaluate pre-deflection analysis results.")

# 3. RESULTS
z0 = -210.75  # z of zero-deflection (note, this is only used to fit 3D shape function below)

# ---

# 4. PROCESS: determine center of disc via peak deflection

# fit bivariate spline to "good" pids during deflection
# alternatively, plot z vs. (x, y) to manually determine peak
deflection_analysis = False
if deflection_analysis:

    px, py = 'frame', 'z'
    frames = df['frame'].unique()

    # meshgrid over the center area
    """
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()
    xspan, yspan = xmax - xmin, ymax - ymin
    xr = np.arange(int(np.round(xmin + xspan / 5)), int(np.round(xmax - xspan / 5) + 1))
    yr = np.arange(int(np.round(ymin + yspan / 5)), int(np.round(ymax - yspan / 5) + 1))
    xv, yv = np.meshgrid(xr, yr)
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()
    """

    # plot
    fr_slices = [end_frames[0] - 75, end_frames[0] - 37, end_frames[0]]
    pxx, pxy = 'x', 'y'
    py = 'z'
    pc = 'id'

    dfrs = df[df['frame'].isin(np.arange(fr_slices[0], fr_slices[-1] + 15))]
    dfrs[py] = (dfrs[py] - z0) * -1
    ylim_min, ylim_max = dfrs[py].min() - 2.5, dfrs[py].max() + 2.5

    fig, axs = plt.subplots(ncols=2, nrows=len(fr_slices), figsize=(10, 2 * len(fr_slices)),
                            sharex=True, sharey=False)

    for i, fr in enumerate(fr_slices):
        dfr = dfrs[dfrs['frame'] == fr]

        axx, axy = axs[i, 0], axs[i, 1]

        axx.scatter(dfr[pxx], dfr[py], c=dfr[pc], s=4, label=fr)
        axy.scatter(dfr[pxy], dfr[py], c=dfr[pc], s=4)

        axx.set_xlim([0, num_pixels])
        # axx.set_ylim([ylim_min, ylim_max])
        axx.grid(alpha=0.25)
        axx.legend(fontsize='small')
        axy.set_xlim([0, num_pixels])
        axy.grid(alpha=0.25)

    axs[-1, 0].set_xlabel(pxx)
    axs[-1, 0].set_ylabel(py)
    axs[-1, 1].set_xlabel(pxy)
    plt.tight_layout()
    plt.savefig(join(path_results_bispl, 'z_by_xy_frame-slice.png'.format(fr)))
    plt.close()