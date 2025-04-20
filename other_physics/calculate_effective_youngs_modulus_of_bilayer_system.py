import numpy as np
import matplotlib.pyplot as plt

def effective_youngs_modulus_bilayer(t_silicone, E_silicone, t_gold, E_gold):
    """
    Computes the effective Young's modulus of a bilayer system (gold on silicone)
    using a rule-of-mixtures type approximation for plane-stress membrane behavior.

    *NOTE: this is the same equation used by Niklaus and Shea (2010) Electrical conductivity
    and Young's modulus of flexible nanocomposites made by metal-ion implantation...

    Parameters:
    - t_silicone: thickness of the silicone layer (in meters)
    - E_silicone: Young's modulus of silicone (in Pascals)
    - t_gold: thickness of the gold film (in meters)
    - E_gold: Young's modulus of gold (in Pascals)

    Returns:
    - E_effective: Effective Young's modulus of the bilayer (in Pascals)
    """

    total_thickness = t_silicone + t_gold
    E_effective = (E_silicone * t_silicone + E_gold * t_gold) / total_thickness
    return E_effective

def effective_youngs_modulus_bilayer_Voigt(t_silicone, E_silicone, t_gold, E_gold):
    """
    Calculate the effective Young's modulus for a bilayer composite material using
    the Voigt model of weighted averages. This model assumes uniform strain across
    the layers and combines the modulus based on the volume fraction of each layer
    in the composite.

    :param t_silicone: Thickness of the silicone layer.
    :type t_silicone: float
    :param E_silicone: Young's modulus of the silicone material.
    :type E_silicone: float
    :param t_gold: Thickness of the gold layer.
    :type t_gold: float
    :param E_gold: Young's modulus of the gold material.
    :type E_gold: float
    :return: The effective Young's modulus of the bilayer composite based on the
             Voigt model.
    :rtype: float
    """
    total_thickness = t_silicone + t_gold
    volume_fraction_silicone = t_silicone / total_thickness
    volume_fraction_gold = t_gold / total_thickness
    E_effective = volume_fraction_silicone * E_silicone + volume_fraction_gold * E_gold
    return E_effective

def effective_youngs_modulus_bilayer_Reuss(t_silicone, E_silicone, t_gold, E_gold):
    """
    Calculates the effective Young's modulus of a bilayer structure
    using the Reuss model, which assumes that the strain is uniform
    across the layer thickness. The effective modulus is computed
    based on the thickness and elastic modulus of each individual layer.

    :param t_silicone: Thickness of the silicone layer
    :type t_silicone: float
    :param E_silicone: Young's modulus of the silicone material
    :type E_silicone: float
    :param t_gold: Thickness of the gold layer
    :type t_gold: float
    :param E_gold: Young's modulus of the gold material
    :type E_gold: float
    :return: Effective Young's modulus of the bilayer material
    :rtype: float
    """
    total_thickness = t_silicone + t_gold
    volume_fraction_silicone = t_silicone / total_thickness
    volume_fraction_gold = t_gold / total_thickness
    E_effective = (E_silicone * E_gold) / (volume_fraction_silicone * E_gold + volume_fraction_gold * E_silicone)
    return E_effective

def effective_youngs_modulus_bilayer_Liu_et_al_2009_transverse_compression(t_A, E_A, nu_A, t_B, E_B, nu_B):
    """
    Calculates the effective Young's modulus (along z-direction, or transverse direction)
    of a bilayer material using the method proposed by Liu et al. (2009). The
    effective Young's modulus is determined accounting for the material properties
    of two layers in the bilayer system, including their thicknesses,
    Young's moduli, and Poisson's ratios.

    NOTE: This assumes transverse compression! Which is NEVER what I am modeling.

    :param t_A: Thickness of the first material layer
    :type t_A: float
    :param E_A: Young's modulus of the first material layer
    :type E_A: float
    :param nu_A: Poisson's ratio of the first material layer
    :type nu_A: float
    :param t_B: Thickness of the second material layer
    :type t_B: float
    :param E_B: Young's modulus of the second material layer
    :type E_B: float
    :param nu_B: Poisson's ratio of the second material layer
    :type nu_B: float
    :return: Effective Young's modulus of the bilayer material
    :rtype: float
    """
    t = t_A + t_B
    V_A = t_A / t
    V_B = t_B / t
    numerator = E_A * E_B
    denominator = V_A * E_B + V_B * E_A - (2 * V_A * V_B * (nu_A * E_B - nu_B * E_A)**2) / ((1-nu_A) * V_B * E_B + (1-nu_B) * V_A * E_A)
    E_effective = numerator / denominator
    return E_effective

def effective_youngs_modulus_bilayer_Liu_et_al_2009(t_A, E_A, nu_A, t_B, E_B, nu_B):
    """
    Calculates the effective Young's modulus (along x-direction, or longitudinal direction)
    of a bilayer material using the method proposed by Liu et al. (2009). The
    effective Young's modulus is determined accounting for the material properties
    of two layers in the bilayer system, including their thicknesses,
    Young's moduli, and Poisson's ratios.

    :param t_A: Thickness of the first material layer
    :type t_A: float
    :param E_A: Young's modulus of the first material layer
    :type E_A: float
    :param nu_A: Poisson's ratio of the first material layer
    :type nu_A: float
    :param t_B: Thickness of the second material layer
    :type t_B: float
    :param E_B: Young's modulus of the second material layer
    :type E_B: float
    :param nu_B: Poisson's ratio of the second material layer
    :type nu_B: float
    :return: Effective Young's modulus of the bilayer material
    :rtype: float
    """
    t = t_A + t_B
    V_A = t_A / t
    V_B = t_B / t
    first_summation = V_A * E_A + V_B * E_B
    numerator = V_A * V_B * E_A * E_B * (nu_A - nu_B)**2
    denominator = V_A * E_A * (1 - nu_B**2) + V_B * E_B * (1 - nu_A**2)
    E_effective = first_summation + numerator / denominator
    return E_effective

def estimate_gold_modulus_from_effective(
    E_eff_measured,
    t_elastomer,
    E_elastomer,
    t_gold
):
    """
    Estimates the Young's modulus of the gold film using a rule-of-mixtures model for in-plane membrane stiffness.

    Parameters:
    - E_eff_measured: effective Young's modulus of the bilayer system (Pa)
    - t_elastomer: thickness of the elastomer layer (m)
    - E_elastomer: Young's modulus of the elastomer (Pa)
    - t_gold: thickness of the gold film (m)

    Returns:
    - E_gold_estimated: estimated Young's modulus of the gold film (Pa)
    """
    t_total = t_elastomer + t_gold
    numerator = E_eff_measured * t_total - E_elastomer * t_elastomer
    E_gold_estimated = numerator / t_gold
    return E_gold_estimated



if __name__ == "__main__":
    SAVE_DIR = '/Users/mackenzie/Library/CloudStorage/Box-Box/2024/Bulge Tests/Thesis'

    # Example parameters
    MEMB_THICKNESS = 20e-6  # 20 microns
    MEMB_YOUNGS_MODULUS = 1.2e6  # 1 MPa
    MEMB_NU = 0.499
    FILM_THICKNESS = 20e-9  # 20 nm
    FILM_YOUNGS_MODULUS = 80e9  # 80 GPa
    FILM_NU = 0.44

    # Compute effective modulus
    E_eff = effective_youngs_modulus_bilayer(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, FILM_THICKNESS, FILM_YOUNGS_MODULUS)
    print(f"Effective Young's modulus: {E_eff:.2e} Pa")

    # Plot E_eff as a function of Au thickness
    au_thicknesses = np.linspace(0, 35e-9, 50)
    E_eff = np.array([effective_youngs_modulus_bilayer(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, x, FILM_YOUNGS_MODULUS) for x in au_thicknesses])
    E_eff_Voigt = np.array(
        [effective_youngs_modulus_bilayer_Voigt(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, x, FILM_YOUNGS_MODULUS) for x in
         au_thicknesses])
    E_eff_Reuss = np.array(
        [effective_youngs_modulus_bilayer_Reuss(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, x, FILM_YOUNGS_MODULUS) for x in
         au_thicknesses])
    E_eff_Liu = np.array(
        [effective_youngs_modulus_bilayer_Liu_et_al_2009(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, MEMB_NU, x, FILM_YOUNGS_MODULUS, FILM_NU) for x in
         au_thicknesses])

    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    ax.plot(au_thicknesses * 1e9, E_eff * 1e-6, 'k-', lw=2, label='Rule-of-Mixtures')
    ax.plot(au_thicknesses * 1e9, E_eff_Voigt * 1e-6, 'r--', lw=2, label='Voigt')
    ax.plot(au_thicknesses * 1e9, E_eff_Reuss * 1e-6, 'b-.', lw=2, label='Reuss')
    ax.plot(au_thicknesses * 1e9, E_eff_Liu * 1e-6, 'g:', lw=2, label='Liu et al. (2009)')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_xlabel("Film Thickness (nm)")
    ax.set_ylabel("Effective Young's Modulus (MPa)")
    ax.grid(alpha=0.25)
    plt.suptitle("Effective Young's Modulus of Silicone-Gold Bilayer")
    ax.set_title("Membrane(E={} MPa, t={} um), Film(Au, E={} GPa)".format(
        MEMB_YOUNGS_MODULUS * 1e-6, MEMB_THICKNESS * 1e6, FILM_YOUNGS_MODULUS * 1e-9),
        fontsize='small')
    plt.tight_layout()
    # plt.savefig(SAVE_DIR + '/E-effective-bilayer_vs_Au-film-thickness_Voigt-and-Liu_XXX.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

    # ----

    # Example usage
    E_eff_measured = 3.81e6  # Pa (measured from bulge test)
    t_elastomer = 20e-6  # m
    E_elastomer = 1.1e6  # Pa
    t_gold = 20e-9  # m

    E_gold_estimated = estimate_gold_modulus_from_effective(
        E_eff_measured,
        t_elastomer,
        E_elastomer,
        t_gold
    )

    print(E_gold_estimated)
