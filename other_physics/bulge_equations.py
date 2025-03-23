from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def pressure_by_deflection_Alaca(w_o, sigma_o, a, t, E, mu):
    return (1 - 0.24 * mu) * (8 / 3) * (E / (1 - mu)) * (t / a ** 4) * w_o**3 + 4 * (sigma_o * t / a**2) * w_o

def pressure_by_deflection_Rosset(w_o, sigma_o, a, t, E, mu):
    return (8 * (1 - 0.24 * mu) * E * t * w_o ** 3) / (3 * (1 - mu) * (a**2 + w_o**2)**2) + (4 * sigma_o * t * a**2 * w_o) / ((a**2 + w_o**2)**2)

def in_plane_strain_Rosset(w_o, a):
    return (2 * w_o**2) / (3 * a**2)

def deflection_for_in_plane_strain_Rosset(strain, a):
    return np.sqrt(3 * strain * a**2 / 2)


def plot_bulge_equations_for_my_test_setup(residual_stress=0, path_save=None):
    # Inputs
    radii = np.array([1, 1.5, 2]) * 1e-3
    t = 20e-6
    w_o = np.linspace(15, 200, 100) * 1e-6

    # Determined by fitting
    Es = np.array([1, 2.5, 5, 10, 25, 75, 100]) * 1e6  # SILPURAN
    mu = 0.5
    sigma_o = residual_stress
    sigma_o_kPa = int(np.round(sigma_o * 1e-3, 2))
    # ---

    # plot
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(4.5 * 2.5, 3.75))

    for a, ax in zip(radii, axes):
        for E in Es:
            p = pressure_by_deflection_Rosset(w_o, sigma_o, a, t, E, mu)
            ax.plot(w_o * 1e6, p, '-', ms=1, label=E * 1e-6)

            ax.set_xlabel(r'$w_o (\mu m)$', fontsize=14)
            ax.set_xticks(np.arange(25, 201, 25))
            ax.grid(alpha=0.35)
            ax.set_title('r={}mm, t={}um'.format(a * 1e3, int(t * 1e6)))

    axes[0].set_ylabel('Pressure (Pa)', fontsize=14)
    # axes[0].set_yscale('log')
    axes[0].set_ylim([-25, 1525])
    axes[0].legend(title='E (MPa)', fontsize='x-small', ncols=4)
    plt.suptitle('Pressure vs. Deflection (' + r'$\sigma_{o}$' + ' = {} kPa)'.format(sigma_o_kPa))
    plt.tight_layout()
    if path_save is not None:
        plt.savefig(
            join(path_save, 'apply_Rosset_2009_to_my-bulge-test_sigma0={}kPa.png'.format(sigma_o_kPa)),
                    dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    SAVE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Modeling'

    residual_stresses = np.array([0, 40, 200, 250, 300, 350, 400]) * 1e3
    for stress in residual_stresses:
        plot_bulge_equations_for_my_test_setup(residual_stress=stress, path_save=SAVE_DIR)