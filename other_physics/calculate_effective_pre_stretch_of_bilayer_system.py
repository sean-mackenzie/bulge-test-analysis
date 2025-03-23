# Define the function to estimate the effect of Au stress on the equilibrium stretch of a bilayer system

def compute_equilibrium_stretch(lambda_initial, t_silicone, E_silicone, t_gold, E_gold, sigma_gold_residual):
    """
    Estimates the effective equilibrium stretch of a silicone-gold bilayer.
    Assumes force balance between stretched silicone and residual-stressed gold film.

    Parameters:
    - lambda_initial: Initial pre-stretch applied to silicone before Au deposition
    - t_silicone: thickness of silicone membrane (in meters)
    - E_silicone: Young's modulus of silicone (in Pascals)
    - t_gold: thickness of gold film (in meters)
    - E_gold: Young's modulus of gold (in Pascals)
    - sigma_gold_residual: residual stress in gold film (in Pascals)

    Returns:
    - lambda_eq: new equilibrium stretch ratio after force balance
    """

    # Biaxial force per unit width in each layer
    N_silicone_initial = E_silicone * t_silicone * (lambda_initial - 1)  # Silicone wants to return to λ=1
    N_gold = sigma_gold_residual * t_gold  # Gold film applies tension (if positive stress)

    # New equilibrium requires total force balance:
    N_silicone_new = N_silicone_initial - N_gold
    lambda_eq = (N_silicone_new / (E_silicone * t_silicone)) + 1

    return lambda_eq

def change_in_pre_stretch(t_silicone, E_silicone, t_gold, sigma_gold_residual):
    return sigma_gold_residual * t_gold / (E_silicone * t_silicone)


if __name__ == "__main__":
    """
    NOTE: 
        * This script estimates the post-deposition equilibrium stretch of a silicone-gold bilayer. 
        * Physical interpretation:
            1. The membrane is initially pre-stretched (e.g., 1.2). 
            2. A metal film is deposited onto the membrane and develops some residual stress. 
            3. The magnitude and state of the residual stress (i.e., compressive or tensile) affects the initial 
            pre-stretch of the membrane, causing it to be at an effective pre-stretch that is more or less than the
            original pre-stretch. 
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Inputs
    MEMB_PRE_STRETCH = 1.25  # initial pre-stretch
    MEMB_THICKNESS = 20e-6  # 20 microns
    MEMB_YOUNGS_MODULUS = 1e6  # ~1 MPa for soft silicone
    FILM_THICKNESS = 20e-9  # 20 nm
    FILM_YOUNGS_MODULUS = 80e9  # ~80 GPa for gold

    # Example and range of film residual stresses
    FILM_RESIDUAL_STRESS = -300e6
    sigma_gold_range = np.linspace(-500e6, 500e6, 200)  # -500 MPa to +500 MPa

    # Calculate equilibrium stretch for each residual stress
    # -- 1. For example residual stress
    delta_pre_stretch = change_in_pre_stretch(MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, FILM_THICKNESS, sigma_gold_residual=FILM_RESIDUAL_STRESS)
    new_equilibrium_stretch = MEMB_PRE_STRETCH + delta_pre_stretch
    # -- 2. For range of residual stresses
    lambda_eq_values = np.array([compute_equilibrium_stretch(MEMB_PRE_STRETCH, MEMB_THICKNESS, MEMB_YOUNGS_MODULUS, FILM_THICKNESS, FILM_YOUNGS_MODULUS, sigma)
                        for sigma in sigma_gold_range])

    # Separate compressive and tensile stress
    sigma_gold_compressive = sigma_gold_range[sigma_gold_range < 0]
    lambda_eq_compressive = lambda_eq_values[sigma_gold_range < 0]

    sigma_gold_tensile = sigma_gold_range[sigma_gold_range >= 0]
    lambda_eq_tensile = lambda_eq_values[sigma_gold_range >= 0]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.axhline(MEMB_PRE_STRETCH, color='gray', linestyle='--', lw=2, label='Initial Pre-Stretch: {}'.format(MEMB_PRE_STRETCH))
    plt.plot(sigma_gold_compressive / 1e6, lambda_eq_compressive,
             color='r', lw=2, label='Effective (Post-Dep) Pre-Stretch: Au compressive stress')
    plt.plot(sigma_gold_tensile / 1e6, lambda_eq_tensile,
             color='b', lw=2, label='Effective (Post-Dep) Pre-Stretch: Au tensile stress')
    plt.title("Effect of Gold Residual Stress on Equilibrium Stretch of Silicone Membrane")
    plt.xlabel("Residual Stress in Gold Film (MPa)")
    plt.ylabel("Equilibrium Stretch Ratio (λ)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
