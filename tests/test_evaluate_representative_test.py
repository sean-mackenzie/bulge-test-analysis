

import os
from os.path import join

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline

from utils import fit, io, plotting

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

base_dir = '/Users/mackenzie/Desktop/Bulge Test/Experiments/BulgeTest_070824_200umSILP-0pT+30nmAu_4mmDia'
read_dir = join(base_dir, 'results/coords')
tid = 6

filename = 'test_coords_testset{}'.format(tid)
save_dir = join(base_dir, 'analyses')
path_results = join(save_dir, 'representative_test{}'.format(tid))
path_results_pids = join(path_results, 'pids')
path_results_fit_plane = join(path_results, 'fit-plane')

for pth in [save_dir, path_results, path_results_pids, path_results_fit_plane]:
    if not os.path.exists(pth):
        os.makedirs(pth)

# 0. experimental parameters
flip_z = True  # if positive pressure, True. If vacuum pressure, False.
scale_z = 1
frame_rate = 32.366
padding = 5
num_pixels = 512
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding

# 1. read
df = io.read_coords(filepath=join(read_dir, filename + '.xlsx'),
                    scale_z=scale_z,
                    z0=0,
                    flip_z=flip_z,
                    frame_rate=frame_rate,
                    skip_frame_zero=True,
                    only_pids=None)

# 2. PROCESS: plot pids to determine "good" pids
start_frame = 50  # average z(frame < start_frame) to estimate dz = 0
end_frames = (150, 300)  # average z(frame > end_frame) to estimate dz_max
eval_pids, plot_pids = True, True
if eval_pids:
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

        y_means.append([pid, start_frame, end_frames[0], end_frames[1],
                        np.round(y_mean_i, 1), np.round(y_mean_f, 1), dy_mean,
                        np.round(dfpid['x'].mean(), 1), np.round(dfpid['y'].mean(), 1),
                        sx, pdeg, rmse_spl, rmse_pf])

        if plot_pids:
            fig, (ax, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

            # plot "raw" 3D particle tracking coords
            ax.plot(dfpid[px], dfpid[py], 'o', ms=1, label=pid, alpha=0.8, zorder=3.5)

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

    df_z_means = pd.DataFrame(np.array(y_means),
                              columns=['pid', 'start_frame', 'end_frame_i', 'end_frame_f',
                                       'z_mean_i', 'z_mean_f', 'dz_mean', 'x', 'y',
                                       'sx', 'pdeg', 'rmse_spl', 'rmse_pf'])
    df_z_means = df_z_means[['pid', 'rmse_pf', 'dz_mean', 'z_mean_i', 'z_mean_f', 'x', 'y',
                             'start_frame', 'end_frame_i', 'end_frame_f',
                             'sx', 'pdeg', 'rmse_spl']]
    df_z_means = df_z_means.sort_values(by='dz_mean', ascending=False)
    df_z_means.to_excel(join(path_results_pids, 'dz-mean_per_pid.xlsx'), index=False)

    # --- plot 2D heat map
    plotting.plot_2D_heatmap(df=df_z_means,
                             pxyz=('x', 'y', 'dz_mean'),
                             savepath=join(path_results_pids, 'dz-mean_per_pid_2D.png'),
                             field=(padding, num_pixels + padding),
                             interpolate='linear',
                             levels=15,
                             )

    # raise ValueError("Pause here to evaluate the results and pick out 'best' and 'bad' pids.")

# NOTES (these variables have no purpose in this script, only "bad_pids")
good_pids = [22, 28, 33, 7, 8, 32, 17, 40, 38]
best_pids = [17]  # z-tracking is excellent (ideally, located at r = 0 (i.e., max deflection)
ok_pids = []  # z-tracking may be useful for analysis
bad_pids = []  # data will be thrown out b/c not useful

# ---

# 3. PROCESS: fit plane to "good" pids before deflection
fit_plane_analysis = True
if fit_plane_analysis:
    px, py = 'frame', 'z'
    frames = df['frame'].unique()

    # not "bad" pids
    df = df[~df['id'].isin(bad_pids)]

    res = []
    for fr in frames:
        dfr = df[df['frame'] == fr]

        # if fr < np.min([start_frame * pre_start, end_frames[0]]):
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
    dfr.to_excel(join(path_results_fit_plane, 'rmse-z__fit-plane-vs-zmean.xlsx'), index=False)

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
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(join(path_results_fit_plane, 'rmse-z__fit-plane-vs-zmean.png'))
    plt.close()

# ---

print("script completed without errors.")