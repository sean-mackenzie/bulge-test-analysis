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

"""
    FN = 'test_coords_test{}'.format(TID)
    SAVE_DIR = join(BASE_DIR, 'analyses')
    path_results = join(SAVE_DIR, 'representative_test{}'.format(TID))
    path_results_pids = join(path_results, 'pids')
    path_results_fit_plane = join(path_results, 'fit-plane')

    for pth in [SAVE_DIR, path_results, path_results_pids, path_results_fit_plane]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    # 0. experimental parameters
    flip_z = True  # if positive pressure, True. If vacuum pressure, False.
    scale_z = 1
    frame_rate = 32.366
    padding = 5
    num_pixels = 512
    img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
"""


if __name__ == "__main__":

    BASE_DIR = '/Users/mackenzie/Desktop/zipper_paper/Testing/Zipper Actuation/01082025_W5-D1_C9-0pT'
    SAVE_DIR = join(BASE_DIR, 'analyses')
    SAVE_COORDS = join(SAVE_DIR, 'coords')
    SAVE_COORDS_W_PIXELS = join(SAVE_COORDS, 'pixels')
    PATH_REPRESENTATIVE = join(SAVE_DIR, 'representative_test{}')

    for pth in [SAVE_DIR, SAVE_COORDS, SAVE_COORDS_W_PIXELS]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    # ---
    # INPUTS
    # ---

    # --- Experimental parameters
    # Optical system
    EXPOSURE_TIME = 0.05  # (s)
    FRAME_RATE = 19.895  # (Hz)
    MICRONS_PER_PIXEL = 1.6  # (microns/pixel)
    IMAGE_SIZE = 512  # number of pixels (assuming size: L x L)
    FIELD_OF_VIEW = IMAGE_SIZE * MICRONS_PER_PIXEL  # microns (L x L)

    # 3D particle tracking
    PADDING = 25  # (pixels)  will be subtracted from raw x-y coordinates
    DROP_FRAME_ZERO = True  # Usually, a faux baseline is used and needs to be removed
    SCALE_Z = -1  # Usually -1 b/c typical direction of calibration and displacement.

    # Relating physical space (microns)
    # to image space (pixels)
    XC_PIXELS, YC_PIXELS, A_PIXELS = 438, 374, 440  # (pixels) position of diaphragm center (i.e., r = 0) and
    # radius, not including padding.
    XC, YC, A = XC_PIXELS * MICRONS_PER_PIXEL, YC_PIXELS * MICRONS_PER_PIXEL, A_PIXELS * MICRONS_PER_PIXEL  # (microns)

    # Relating external stimulus (time-dependent voltage)
    # to image space (frames)
    START_FRAME = 90  # (frames) when the particles started moving
    END_FRAMES = (165, 300)  # when particles were at maximum displacement

    # ---
    # PRE-PROCESSING
    # ---

    # Pre-process particle coordinates
    READ_DIR = join(BASE_DIR, 'results', 'test-idpt_test-{}')
    FN_STARTS_WITH = 'test_coords_t'
    TID  = 2
    # -
    PRE_PROCESS_COORDS = False
    if PRE_PROCESS_COORDS:
        # Find and read coords
        FN = [x for x in os.listdir(READ_DIR.format(TID)) if x.startswith(FN_STARTS_WITH)]
        DF = pd.read_excel(join(READ_DIR.format(TID), FN[0]))

        # Format the coords
        KEEP_COLS = ['frame', 'id', 'cm', 'xg', 'yg', 'z']
        DICT_RENAME = {'xg': 'x_pix', 'yg': 'y_pix'}
        DF = DF[KEEP_COLS].rename(columns=DICT_RENAME)

        # Remove 3D particle tracking artifacts
        # scale z
        DF['z'] = DF['z'] * SCALE_Z
        # remove padding
        DF['x_pix'] = DF['x_pix'] - PADDING
        DF['y_pix'] = DF['y_pix'] - PADDING
        # remove faux baseline
        if DROP_FRAME_ZERO:
            DF = DF[DF['frame'] != 0]

        # Relate image coordinates to image features
        # relative to r = 0 (i.e., center of diaphragm)
        DF['r_pix'] = np.sqrt((DF['x_pix'] - XC_PIXELS) ** 2 + (DF['y_pix'] - YC_PIXELS) ** 2)

        # Transform coordinates from image space (frames, pixels) to physical space (seconds, microns)
        # frames-to-seconds
        DF['t'] = DF['frame'] / FRAME_RATE  # (s)
        # pixels-to-microns
        DF['x'] = DF['x_pix'] * MICRONS_PER_PIXEL  # (microns)
        DF['y'] = DF['y_pix'] * MICRONS_PER_PIXEL  # (microns)
        DF['r'] = np.sqrt((DF['x'] - XC) ** 2 + (DF['y'] - YC) ** 2)

        # Define relative positions (i.e., displacement)
        # relative to: t < t_start
        DF0 = DF[DF['frame'] < START_FRAME].groupby('id').mean()
        DFPIDS = []
        for pid in DF['id'].unique():
            DFPID = DF[DF['id'] == pid]
            # time
            DFPID['dt'] = DFPID['t'] - START_FRAME / FRAME_RATE
            # space
            for v in ['x', 'y', 'r', 'z', 'x_pix', 'y_pix', 'r_pix']:
                DFPID['d' + v] = DFPID[v] - DF0.iloc[pid][v]
            DFPIDS.append(DFPID)
        DF = pd.concat(DFPIDS)

        # Export transformed dataframe
        # for reference: physical + pixel coordinates
        DF.to_excel(join(SAVE_COORDS_W_PIXELS, '{}{}.xlsx'.format(FN_STARTS_WITH, TID)), index=False)
        # for analysis: we need only keep physical coordinates (and frames) for compactness
        PHYS_COLS = ['frame', 't', 'id', 'dx', 'dy', 'dr', 'dz', 'x', 'y', 'r', 'z']
        DF = DF[PHYS_COLS]
        DF.to_excel(join(SAVE_COORDS, '{}{}.xlsx'.format(FN_STARTS_WITH, TID)), index=False)
    else:
        DF = pd.read_excel(join(SAVE_COORDS, '{}{}.xlsx'.format(FN_STARTS_WITH, TID)))


    # ---
    # FIRST-PASS EVALAUATION
    # ---

    # Plot particle trajectories as first-pass outlier rejection scheme
    PLOT_COORDS = True

    if PLOT_COORDS:

        # --- THREE FUNCTIONS: plot each pid, calculate net displacement, plot 2D heat map
        # -
        # function: inputs
        eval_pids_drz = True  # True: calculate net-displacement per-particle in r- and z-directions
        plot_pids = False  # True: plot particle z-trajectories
        plot_heatmaps = True  # True: plot 2D heat map (requires eval_pids_dz having been run)
        eval_pids_dz = False  # To be deprecated

        df = DF
        py = 'dz'  # options: 'z': position; 'dz': displacement
        pr = 'dr'  # options: 'r': position; 'dr': displacement
        start_frame = START_FRAME  # average z(frame < start_frame) to estimate dz = 0
        end_frames = END_FRAMES  # average z(frame > end_frame) to estimate dz_max
        path_results_pids = PATH_REPRESENTATIVE.format(TID)
        field_of_view = FIELD_OF_VIEW  # for plotting 2D heat map
        sx = 0.25  # fitting parameter: smoothing factor, s = sx * (number of samples to fit)
        pdeg = 12  # fitting parameter: degree of polynominal
        # -
        # function: routine
        # initialize variables
        df_z_means = None
        df_rz_means = None
        # make directory
        if not os.path.exists(path_results_pids):
            os.makedirs(path_results_pids)

        if eval_pids_dz:
            px = 'frame'
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
                x = dfpid[(dfpid[px] > start_frame) & (dfpid[px] < end_frames[0])][px].to_numpy()
                y = dfpid[(dfpid[px] > start_frame) & (dfpid[px] < end_frames[0])][py].to_numpy()

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
                    fig, (ax, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
                                                  gridspec_kw={'height_ratios': [2, 1]})

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

        if eval_pids_drz:
            px = 'frame'
            pids = df['id'].unique()
            pids.sort()

            y_means = []
            for pid in pids:
                dfpid = df[df['id'] == pid]

                # initial r-z position
                x_pos_mean_i = dfpid[dfpid[px] < start_frame]['x'].mean()
                y_pos_mean_i = dfpid[dfpid[px] < start_frame]['y'].mean()
                r_pos_mean_i = dfpid[dfpid[px] < start_frame]['r'].mean()
                z_pos_mean_i = dfpid[dfpid[px] < start_frame]['z'].mean()

                # dr analysis
                r_mean_i = dfpid[dfpid[px] < start_frame][pr].mean()
                r_mean_f = dfpid[(dfpid[px] > end_frames[0]) & (dfpid[px] < end_frames[1])][pr].mean()
                dr_mean = np.round(r_mean_f - r_mean_i, 2)

                # dz analysis
                y_mean_i = dfpid[dfpid[px] < start_frame][py].mean()
                y_mean_f = dfpid[(dfpid[px] > end_frames[0]) & (dfpid[px] < end_frames[1])][py].mean()
                dy_mean = np.round(y_mean_f - y_mean_i, 2)

                y_means.append([pid,
                                np.round(x_pos_mean_i, 1),
                                np.round(y_pos_mean_i, 1),
                                np.round(r_pos_mean_i, 1),
                                np.round(z_pos_mean_i, 1),
                                np.round(r_mean_i, 2),
                                np.round(y_mean_i, 1),
                                np.round(r_mean_f, 2),
                                np.round(y_mean_f, 1),
                                dr_mean,
                                dy_mean,
                                np.round((dr_mean + dfpid['r'].mean()) / dfpid['r'].mean(), 4),
                                np.round(dr_mean / dy_mean, 4),
                                start_frame,
                                end_frames[0],
                                end_frames[1],
                                ])

                if plot_pids:
                    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True,
                                                  gridspec_kw={'height_ratios': [1.25, 1]})

                    # plot z/dz by frames
                    ax1.plot(dfpid[px], dfpid[py], '-o', ms=1.5, lw=0.75, label=pid)
                    ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
                    ax1.legend(title=r'$p_{ID}$')
                    ax1.grid(alpha=0.25)
                    ax1.set_title(r'$\Delta z_{net}=$' + ' {} '.format(dy_mean) + r'$\mu m$')

                    # plot r/dr by frames
                    ax2.plot(dfpid[px], dfpid[pr], '-o', ms=1.5, lw=0.75)
                    ax2.set_xlabel('Frame')
                    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                    ax2.grid(alpha=0.25)
                    ax2.set_title(r'$\Delta r_{net}=$' + ' {} '.format(dr_mean) + r'$\mu m$')

                    plt.tight_layout()
                    plt.savefig(join(path_results_pids, 'pid{}.png'.format(pid)), dpi=300, facecolor='w')
                    plt.close()

            df_rz_means = pd.DataFrame(np.array(y_means),
                                      columns=['pid', 'x', 'y', 'r', 'z',
                                               'dr_mean_i', 'dz_mean_i',
                                               'dr_mean_f', 'dz_mean_f',
                                               'dr_mean', 'dz_mean',
                                               'r_strain', 'drdz',
                                               'start_frame', 'end_frame_i', 'end_frame_f',
                                               ])
            df_rz_means = df_rz_means.sort_values(by='dz_mean', ascending=False)
            df_rz_means.to_excel(join(path_results_pids, 'dzr-mean_per_pid.xlsx'), index=False)

        # --- plot 2D heat map
        if plot_heatmaps:
            if df_rz_means is not None:
                plotting.plot_2D_heatmap(df=df_rz_means,
                                         pxyz=('x', 'y', 'dz_mean'),
                                         savepath=join(path_results_pids, 'dz-mean_per_pid_2D.png'),
                                         field=(0, field_of_view),
                                         interpolate='linear',
                                         levels=15,
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta z \: (\mu m)$')
                                         )
                plotting.plot_2D_heatmap(df=df_rz_means,
                                         pxyz=('x', 'y', 'dr_mean'),
                                         savepath=join(path_results_pids, 'dr-mean_per_pid_2D.png'),
                                         field=(0, field_of_view),
                                         interpolate='linear',
                                         levels=15,
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r \: (\mu m)$'),
                                         )

                plotting.plot_2D_heatmap(df=df_rz_means,
                                         pxyz=('x', 'y', 'r_strain'),
                                         savepath=join(path_results_pids, 'dr-strain_per_pid_2D.png'),
                                         field=(0, field_of_view),
                                         interpolate='linear',
                                         levels=15,
                                         units=(r'$(\mu m)$', r'$(\mu m)$', r'$\Delta r / r$'),
                                         )




    a = 1