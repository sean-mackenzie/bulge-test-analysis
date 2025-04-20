from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import itertools


# Set marker cycler
# markers = ['o', 's', 'D', '^', 'v']  # Define marker styles
# plt.rc('axes', prop_cycle=cycler(marker=markers))  # Apply marker cycle globally


from other_physics.bulge_equations import pressure_by_deflection_Rosset
from other_physics.calculate_thickness_after_pre_stretch import calculate_stretched_thickness


# ------------------------ HELPER FUNCTIONS

def fit_line(x, a, b):
    return a * x + b


# ------------------------ OLD PLOTTING FUNCTION

def plot_bulge_data_overlay_on_equations():
    # Define typical parameters
    t = 20e-6
    w_o = np.linspace(2, 225, 100) * 1e-6
    E_MPa = 3.25  # np.array([1, 5, 10, 50]) * 1e6
    mu = 0.5
    sigma_o_kPa = np.array([28])

    # read bulge test data: pressure vs. deflection
    df = pd.read_excel(FP_ALL_PZ)
    # df = df[(df['pre_stretch_measured'] > 12) & (df['pre_stretch_measured'] < 18)]
    df = df[df['bta_id'] == '20250104_C9-0pT_20nmAu_4mmDia']

    pre_stretch_model_ = 0.0  # df['pre_stretch_measured'].mean()
    pre_stretch_model = np.round(1 + pre_stretch_model_ / 100, 3)
    thickness_post_stretch = calculate_stretched_thickness(original_thickness=t, stretch_factor=pre_stretch_model)
    thickness_post_stretch_ = np.round(thickness_post_stretch * 1e6, 2)

    # SAVE_DIR = join(SAVE_DIR, 'E={}MPa'.format(E_MPa))
    # if not os.path.exists(SAVE_DIR):
    #    os.makedirs(SAVE_DIR)

    # iterate
    radii = df['radius_mm'].unique()
    radii = [2.0]
    # plot
    for r in radii:
        df_r = df[df['radius_mm'] == r]

        fig, ax = plt.subplots(figsize=(8, 6))
        for bta_id in df_r.sort_values('pre_stretch_measured')['bta_id'].unique():
            df_bta = df_r[df_r['bta_id'] == bta_id]

            # get only data for tid of max pressure
            tid_of_max_pressure = df_bta['tid'].iloc[np.argmax(df_bta['P'])]
            df_bta_max_pressure = df_bta[df_bta['tid'] == tid_of_max_pressure]

            lbl_memb_id = df_bta_max_pressure['memb_id'].iloc[0]
            lbl_deposit_Au = df_bta_max_pressure['deposit_Au'].iloc[0]
            lbl_pre_stretch_measured = df_bta_max_pressure['pre_stretch_measured'].iloc[0]

            ax.plot(df_bta_max_pressure['z'], df_bta_max_pressure['P'], '-o', ms=2,
                    label='{}: {}pT + {}nmAu'.format(lbl_memb_id, lbl_pre_stretch_measured, lbl_deposit_Au))

        for sigma_o in sigma_o_kPa:
            p = pressure_by_deflection_Rosset(w_o, sigma_o * 1e3, r * 1e-3, thickness_post_stretch, E_MPa * 1e6, mu)
            ax.plot(w_o * 1e6, p, '-', lw=1, label='Rossett (E={} MPa, sigma_o={} kPa)'.format(E_MPa, sigma_o))

        ax.set_xlabel('Deflection (um)')
        ax.set_ylabel('Pressure (Pa)')
        ax.grid(alpha=0.25)
        ax.legend(loc='best', fontsize='small', markerscale=3)  # , bbox_to_anchor=(1, 1)
        ax.set_title('Radius = {} mm: Pre-stretch={}, Post-stretch-thickness={}um'.format(r, pre_stretch_model,
                                                                                          thickness_post_stretch_))

        ax.set_ylim([-10, 240])
        if r == 1.0:
            ax.set_ylim([-25, 625])
            ax.set_xlim([-2.5, 32.5])

        plt.tight_layout()
        plt.savefig(join(SAVE_DIR, 'E={}MPa_tmemb-PS={}um_r={}mm.png'.format(E_MPa, thickness_post_stretch_, r)),
                    dpi=300, facecolor='w', bbox_inches='tight')
        plt.show()
        plt.close()

# ------------------------ PARAMETERIZED PLOTTING FUNCTIONS

def plot_parameterized_bulge_test_results():
    pass


def plot_E_sigma0_by_px(df, px, px_label, path_save=None, save_id=None, title=None, include_legend=False):
    # Define columns to plot
    pys = ['l_E_MPa', 'nl_E_MPa', 'l_s0_MPa', 'nl_s0_MPa']
    # Attach a generic handle to save_id
    save_id = save_id + f'_E-sigma0_by_{px}_scatter'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 5),
                                gridspec_kw={'width_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # plot data for each bulge test
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.plot(df_bta[px], df_bta[py], marker=mrk, ms=5, ls='none', label=bta_id)
    # format figure
    for i in range(2):
        axs[1, i].set_xlabel(px_label)
        axs[0, i].grid(alpha=0.25)
        axs[1, i].grid(alpha=0.25)
    axs[0, 0].set_ylabel("$E_{bta-output}$ (MPa)")
    axs[1, 0].set_ylabel("$\sigma_{o,bta-output}$ (MPa)")
    axs[0, 0].set_title('Linear model')
    axs[0, 1].set_title('Nonlinear model')
    if include_legend:
        axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def errorbars_E_sigma0_by_px(df, px, px_label, path_save=None, save_id=None, title=None, include_legend=False, xerr=0.0):
    # Define columns to plot
    pys = ['l_E_MPa', 'nl_E_MPa', 'l_s0_MPa', 'nl_s0_MPa']
    # Attach a generic handle to save_id
    save_id = save_id + f'_E-sigma0_by_{px}_errorbars'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 5),
                                gridspec_kw={'width_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # plot data for each bulge test
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.errorbar(df_bta[px].mean(), df_bta[py].mean(), yerr=df_bta[py].std(), xerr=xerr,
                    fmt=mrk, ms=5, ls='none', capsize=2, elinewidth=1, label=bta_id)
    # format figure
    for i in range(2):
        axs[1, i].set_xlabel(px_label)
        axs[0, i].grid(alpha=0.25)
        axs[1, i].grid(alpha=0.25)
    axs[0, 0].set_ylabel("$E_{bta-output}$ (MPa)")
    axs[1, 0].set_ylabel("$\sigma_{o,bta-output}$ (MPa)")
    axs[0, 0].set_title('Linear model')
    axs[0, 1].set_title('Nonlinear model')
    if include_legend:
        axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_E_sigma0_by_pre_stretch(df, path_save=None, save_id=None, include_legend=False):
    # Define columns to plot
    px = 'ps_bta'
    pys = ['l_E_MPa', 'nl_E_MPa', 'l_s0_MPa', 'nl_s0_MPa']
    # Attach a generic handle to save_id
    save_id = save_id + '_E-sigma0_by_pre_stretch_scatter'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 5),
                                gridspec_kw={'width_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # plot data for each bulge test
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.plot(df_bta[px], df_bta[py], marker=mrk, ms=5, ls='none', label=bta_id)
    # format figure
    for i in range(2):
        axs[1, i].set_xlabel(r'$\lambda_{bta-input}$')
        axs[1, i].set_xticks(np.arange(1.0, 1.275, 0.05))
        axs[0, i].grid(alpha=0.25)
        axs[1, i].grid(alpha=0.25)
    axs[0, 0].set_ylabel("$E_{bta-output}$ (MPa)")
    axs[1, 0].set_ylabel("$\sigma_{o,bta-output}$ (MPa)")
    axs[0, 0].set_title('Linear model')
    axs[0, 1].set_title('Nonlinear model')
    if include_legend:
        axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def errorbars_E_sigma0_by_pre_stretch(df, path_save=None, save_id=None, include_legend=False, xerr=0.01):
    # Define columns to plot
    px = 'ps_bta'
    pys = ['l_E_MPa', 'nl_E_MPa', 'l_s0_MPa', 'nl_s0_MPa']
    # Attach a generic handle to save_id
    save_id = save_id + '_E-sigma0_by_pre_stretch_errorbars'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 5),
                                gridspec_kw={'width_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # plot data for each bulge test
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.errorbar(df_bta[px].mean(), df_bta[py].mean(), yerr=df_bta[py].std(), xerr=xerr,
                    fmt=mrk, ms=5, ls='none', capsize=2, elinewidth=1, label=bta_id)
    # format figure
    for i in range(2):
        axs[1, i].set_xlabel(r'$\lambda_{bta-input}$')
        axs[1, i].set_xticks(np.arange(1.0, 1.275, 0.05))
        axs[0, i].grid(alpha=0.25)
        axs[1, i].grid(alpha=0.25)
    axs[0, 0].set_ylabel("$E_{bta-output}$ (MPa)")
    axs[1, 0].set_ylabel("$\sigma_{o,bta-output}$ (MPa)")
    axs[0, 0].set_title('Linear model')
    axs[0, 1].set_title('Nonlinear model')
    if include_legend:
        axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def fit_sigma0_by_pre_stretch(df, path_save=None, save_id=None, include_legend=False):
    # Define columns to plot
    px = 'ps_bta'
    pys = ['l_s0_MPa', 'nl_s0_MPa']
    lbls = ['Linear model', 'Nonlinear model']
    # Sort by independent variable
    df = df.sort_values(px)
    # Attach a generic handle to save_id
    save_id = save_id + '_sigma0_by_pre_stretch_fit'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(6, 5),
                                gridspec_kw={'height_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # fit line
    for ax, py, lbl in zip(axes, pys, lbls):
        xdata = df[px]
        ydata = df[py]
        popt, pcov = np.polyfit(xdata, ydata, 1, cov=True)
        p = np.poly1d(popt)
        yfit = p(xdata)
        ax.plot(xdata, yfit, '--', color='k', lw=0.75)
        # ax.plot(xdata, ydata, 'o', ms=3, color='k', alpha=0.5)
        ax.set_title('{}: y = {:.2f}x + {:.2f}'.format(lbl, popt[0], popt[1]))
        ax.set_ylabel('$ \sigma_{o, bta-output}$ (MPa)')
        ax.grid(alpha=0.25)
    # plot each bulge test data
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.plot(df_bta[px], df_bta[py], marker=mrk, ms=5, ls='none', label=bta_id)
    # format figure
    axs[1].set_xlabel(r'$\lambda_{bta-input}$')
    axs[1].set_xticks(np.arange(1.0, df[px].max() + 0.025, 0.05))
    if include_legend:
        axs[0].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def fit_errorbars_sigma0_by_pre_stretch(df, path_save=None, save_id=None, include_legend=False, xerr=0.01):
    # Define columns to plot
    px = 'ps_bta'
    pys = ['l_s0_MPa', 'nl_s0_MPa']
    lbls = ['Linear model', 'Nonlinear model']
    # Sort by independent variable
    df = df.sort_values(px)
    # Attach a generic handle to save_id
    save_id = save_id + '_sigma0_by_pre_stretch_fit_errorbars'
    # Define figure size to accommodate legend
    if include_legend:
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(6, 5),
                                gridspec_kw={'height_ratios': [1, 1]})
        save_id = save_id + '_with_legend'
    else:
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4, 5))
    axes = axs.flatten()
    # Define markers in a cycle
    marker_cycle = itertools.cycle(['o', 's', 'd', '^', 'v', '*', 'P', 'X', 'D'])
    # ---
    # fit line
    for ax, py, lbl in zip(axes, pys, lbls):
        xdata = df[px]
        ydata = df[py]
        popt, pcov = np.polyfit(xdata, ydata, 1, cov=True)
        p = np.poly1d(popt)
        yfit = p(xdata)
        ax.plot(xdata, yfit, '--', color='k', lw=0.75)
        # ax.plot(xdata, ydata, 'o', ms=3, color='k', alpha=0.5)
        ax.set_title('{}: y = {:.2f}x + {:.2f}'.format(lbl, popt[0], popt[1]))
        ax.set_ylabel('$ \sigma_{o, bta-output}$ (MPa)')
        ax.grid(alpha=0.25)
    # plot each bulge test data
    bta_ids = df['bta_id'].unique()
    for bta_id in bta_ids:
        df_bta = df[df['bta_id'] == bta_id]
        mrk = next(marker_cycle)
        for ax, py in zip(axes, pys):
            ax.errorbar(df_bta[px].mean(), df_bta[py].mean(), yerr=df_bta[py].std(), xerr=xerr,
                        fmt=mrk, ms=5, ls='none', capsize=2, elinewidth=1, label=bta_id)
    # format figure
    axs[1].set_xlabel(r'$\lambda_{bta-input}$')
    axs[1].set_xticks(np.arange(1.0, df[px].max() + 0.025, 0.05))
    if include_legend:
        axs[0].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        axs[1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(join(path_save, save_id + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # ---

    BASE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/AnalysesCombined'
    READ_DIR = BASE_DIR
    SAVE_DIR = join(BASE_DIR, 'relationships')
    # -
    FP_ALL_PZ = join(READ_DIR, 'all-bulge-test-analyses-PZ-by-t-by-tid.xlsx')
    FP_ALL_OVERVIEW = join(READ_DIR, 'all-bulge-test-analyses-results-overview.xlsx')
    DF = pd.read_excel(FP_ALL_OVERVIEW)
    # -
    # -
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # -
    # -
    # --- 1. COMPARE ALL BULGE TEST DATA ACROSS APPLICABLE VARIABLES
    # In this case, the only generally applicable variable is pre_stretch

    compare_all = False
    if compare_all:
        SAVE_ID = 'compare-all'
        EST_PS_UNCERTAINTY = 0.015

        SAVE_SUB_DIR = join(SAVE_DIR, SAVE_ID)
        if not os.path.exists(SAVE_SUB_DIR):
            os.makedirs(SAVE_SUB_DIR)

        for legend in [True, False]:
            plot_E_sigma0_by_pre_stretch(DF, path_save=SAVE_SUB_DIR, save_id=SAVE_ID, include_legend=legend)
            errorbars_E_sigma0_by_pre_stretch(DF, path_save=SAVE_SUB_DIR, save_id=SAVE_ID, include_legend=legend, xerr=EST_PS_UNCERTAINTY)
            fit_sigma0_by_pre_stretch(DF, path_save=SAVE_SUB_DIR, save_id=SAVE_ID, include_legend=legend)
            fit_errorbars_sigma0_by_pre_stretch(DF, path_save=SAVE_SUB_DIR, save_id=SAVE_ID, include_legend=legend, xerr=EST_PS_UNCERTAINTY)
    # -
    # -
    # --- 2. COMPARE SUBSETS OF BULGE TEST DATA ACROSS APPLICABLE VARIABLES
    # This includes: t_Au(ps_meas==1), t_Au(1.2 <= ps_meas < 1.3), t_Au(1.1 < ps_meas < 1.15),

    compare_subsets = True
    if compare_subsets:
        # Define local variables
        df = DF
        save_dir = SAVE_DIR
        # Define subsets
        ps_subset = ['ps_bta', 'ps_bta', 'ps_bta']
        ps_limits = [(0.9, 1.01), (1.19, 1.3), (1.1, 1.15)]
        ps_save_ids = ['no-pre-stretch', 'high-pre-stretch', 'med-pre-stretch']
        ps_titles = [r'$\lambda = 1$', r'$1.2 \leq \lambda < 1.3$', r'$1.1 < \lambda < 1.15$']
        # Define independent variables
        pxs = ['t_Au', 't_Au', 't_Au']
        px_labels = [r'$t_{Au} \: (nm)$', r'$t_{Au} \: (nm)$', r'$t_{Au} \: (nm)$']
        px_xerrs = [0.0, 0.0, 0.0]
        # Define shared modifiers
        legend = False
        # -
        # iterate through subsets
        for i in range(len(ps_subset)):
            # create sub-directory
            save_id = ps_save_ids[i]
            save_sub_dir = join(save_dir, save_id)
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            # get subset
            df_subset = df[(df[ps_subset[i]] >= ps_limits[i][0]) & (df[ps_subset[i]] < ps_limits[i][1])]
            # plot
            for legend in [True, False]:
                plot_E_sigma0_by_px(df_subset, px=pxs[i], px_label=px_labels[i],
                                    path_save=save_sub_dir, save_id=save_id,
                                    title=ps_titles[i], include_legend=legend)
                errorbars_E_sigma0_by_px(df_subset, px=pxs[i], px_label=px_labels[i],
                                         path_save=save_sub_dir, save_id=save_id,
                                         title=ps_titles[i], include_legend=legend, xerr=px_xerrs[i])



