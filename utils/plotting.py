import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
"""
See more on backends: https://matplotlib.org/stable/users/explain/figure/backends.html
Another option: https://stackoverflow.com/questions/49048520/how-to-prevent-pycharm-from-overriding-default-backend-as-set-in-matplotlib
"""


def show_P_by_dt(df, savepath=None):
    fig, ax = plt.subplots()
    ax.plot(df['dt'], df['P'], '-o', ms=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressue (Pa)')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()


def show_combined_P_by_dt(dfs, savepath=None):
    """
    plotting.show_combined_P_by_dt(dfs)
    :param dfs:
    :return:
    """
    groups = dfs['group_reset'].unique()
    num_groups = len(groups)
    if num_groups > 1:
        fig, axs = plt.subplots(ncols=len(groups), figsize=(5.5 * num_groups, 4.25))
    else:
        fig, axs = plt.subplots()
        axs = [axs]
    for g, ax in zip(groups, axs):
        dfsg = dfs[dfs['group_reset'] == g]
        marker = iter(['o', 'x', 'd', '^', 's'])
        m, i = next(marker), 0
        for tid in dfsg['tid'].unique():
            df = dfsg[dfsg['tid'] == tid]
            i += 1
            if i > 10:
                i = 0
                m = next(marker)
            ax.plot(df['t_reset'] / 1000, df['P'], ms=1, marker=m, label=tid)
        ax.set_xlabel(r'$t_{reset} \: (s)$')
        ax.set_ylabel('P (Pa)')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title('Group since reset: {}'.format(g))
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()


def show_z_by_dt(df, savepath=None):
    fig, ax = plt.subplots()
    for pid in df['id'].unique():
        dfpid = df[df['id'] == pid]
        ax.plot(dfpid['dt'], dfpid['z'], '-o', ms=2, label=pid)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.legend(title=r'$p_{ID}$')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()


def show_combined_z_by_dt(dfs, savepath=None):
    fig, ax = plt.subplots()
    tids = dfs['tid'].unique()
    for tid in tids:
        dftid = dfs[dfs['tid'] == tid].groupby('dt').mean().reset_index()
        ax.plot(dftid['dt'], dftid['z'], '-o', ms=2,
                label='{}: {}'.format(tid, np.round(dftid['z'].max(), 1)))
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.arange(0, dfs['dt'].max()))
    ax.set_ylabel(r'$z_{mean}(t) \: (\mu m)$')
    ax.set_yticks(np.arange(0, dfs['z'].max() + 3.5, 5))
    ax.legend(title=r'$test_{ID}: \: \Delta_{max} z \: (\mu m)$')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()