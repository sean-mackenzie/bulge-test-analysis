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

    # Example parameters
    MEMB_THICKNESS = 20e-6  # 20 microns
    MEMB_YOUNGS_MODULUS = 1e6  # 1 MPa
    FILM_THICKNESS = 20e-9  # 20 nm
    FILM_YOUNGS_MODULUS = 80e9  # 80 GPa

    # Compute effective modulus
    E_eff = effective_youngs_modulus_bilayer(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, FILM_THICKNESS, FILM_YOUNGS_MODULUS)
    print(f"Effective Young's modulus: {E_eff:.2e} Pa")

    # Plot E_eff as a function of Au thickness
    au_thicknesses = np.linspace(0, 35e-9, 50)
    E_eff = np.array([effective_youngs_modulus_bilayer(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, x, FILM_YOUNGS_MODULUS) for x in au_thicknesses])
    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    ax.plot(au_thicknesses * 1e9, E_eff * 1e-6, color='k', lw=2)
    ax.set_xlabel("Film Thickness (nm)")
    ax.set_ylabel("Effective Young's Modulus (MPa)")
    ax.grid(alpha=0.25)
    plt.suptitle("Effective Young's Modulus of Silicone-Gold Bilayer")
    ax.set_title("Membrane(E = {} MPa, t = {} um), Film(Au, E = {} GPa)".format(
        MEMB_YOUNGS_MODULUS * 1e-6, MEMB_THICKNESS * 1e6, FILM_YOUNGS_MODULUS * 1e-9),
        fontsize='small')
    plt.tight_layout()
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
