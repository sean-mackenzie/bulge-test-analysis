import pandas as pd
from utils import io

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def replace_with_extrapolated_pressure(dfs, extrapolate_tid, fit_dt, extrapolate_to, savepath):
    # 0. store raw data
    dfs_orig = dfs.copy()

    # 1. get data for this tid
    dfs_fit = dfs[dfs['tid'] == extrapolate_tid]

    # 2. subset data within time bounds (where time is proxy for pressure)
    df = dfs_fit[(dfs_fit['dt'] > fit_dt[0]) & (dfs_fit['dt'] < fit_dt[1])]

    # 3. fit line
    fx = df['dt'].to_numpy()
    fy = df['P'].to_numpy()
    def fit_line(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(fit_line, fx, fy)
    a, b = popt

    # --- 4. replace data with extrapolated data
    # 4a. get extrapolated data for specified range
    df_extrapolate = dfs_fit[dfs_fit['dt'] > fit_dt[1]]
    xnew = df_extrapolate['dt'].to_numpy()
    ynew = fit_line(xnew, a, b)
    x_extrapolate_to = xnew[ynew < extrapolate_to]
    y_extrapolate_to = ynew[ynew < extrapolate_to]
    # 4b. replace old data with extrapolated data
    dfs.loc[(dfs['tid'] == extrapolate_tid) &
            (dfs['dt'] > fit_dt[1]) &
            (dfs['dt'] <= x_extrapolate_to.max()), 'P'] = y_extrapolate_to

    # --- plot
    df_extrapolate['P_new'] = df_extrapolate['P']
    df_extrapolate.loc[df_extrapolate['dt'] <= x_extrapolate_to.max(), 'P_new'] = y_extrapolate_to

    fig, ax = plt.subplots()

    ax.plot(dfs_orig[dfs_orig['tid'] == extrapolate_tid]['dt'],
            dfs_orig[dfs_orig['tid'] == extrapolate_tid]['P'],
            'o', ms=1, color='gray', label='raw')

    ax.plot(fx, fy, marker='d', ms=2, linestyle='none', color='k', label='data')
    ax.plot(fx, a * fx + b, '-', lw=0.75, color='r', label='fit')

    ax.plot(df_extrapolate['dt'], df_extrapolate['P_new'], marker='s', ms=1,
            linestyle='none', color='tab:green', label='extrapolated')

    ax.grid(alpha=0.25)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')

    return dfs