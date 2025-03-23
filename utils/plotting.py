import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

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


def show_combined_P_by_dt(dfs, savepath=None, suptitle=None):
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

    if suptitle is not None:
        plt.suptitle(suptitle)

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
    # ax.set_yticks(np.arange(0, dfs['z'].max() + 3.5, 5))
    ax.legend(title=r'$test_{ID}: \: \Delta_{max} z \: (\mu m)$')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()


# ---

def plot_2D_heatmap(df, pxyz, savepath=None, field=None, interpolate='linear', levels=15, units=None, title=None):
    """

    :param df:
    :param pxyz:
    :param: savepath:
    :param field: (0, side-length of field-view (pixels or microns))
    :param interpolate:
    :param levels:
    :param units: two-tuple (x-y units, z units), like: ('pixels', r'$\Delta z \: (\mu m)$')
    :return:
    """

    # get data
    x, y, z = df[pxyz[0]].to_numpy(), df[pxyz[1]].to_numpy(), df[pxyz[2]].to_numpy()

    # if no field is passed, use x-y limits.
    if field is None:
        field = (np.min([x.min(), y.min()]), np.max([x.max(), y.max()]))
    # if no units, don't assume any
    if units is None:
        units = ('', '', '')

    # Create grid values.
    xi = np.linspace(field[0], field[1], len(df))
    yi = np.linspace(field[0], field[1], len(df))

    # interpolate
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=interpolate)

    # plot
    fig, ax = plt.subplots()

    ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")

    fig.colorbar(cntr1, ax=ax, label=units[2])
    ax.plot(x, y, 'ko', ms=3)
    ax.set(xlim=(field[0], field[1]), xticks=(field[0], field[1]),
           ylim=(field[0], field[1]), yticks=(field[0], field[1]),
           )
    ax.invert_yaxis()
    ax.set_xlabel(r'$x$ ' + units[0])
    ax.set_ylabel(r'$y$ ' + units[1])

    if title is not None:
        ax.set_title(title)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor='white')
    else:
        plt.show()
    plt.close()

#