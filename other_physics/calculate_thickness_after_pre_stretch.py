def calculate_stretched_thickness(original_thickness, stretch_factor):
    """
    Calculate the thickness of a membrane after biaxial stretching.

    Parameters:
        original_thickness (float): The original thickness of the membrane in microns (or any unit).
        stretch_factor (float): The biaxial stretch factor (e.g., 1.2 for 20% stretching).

    Returns:
        float: The thickness of the stretched membrane in the same unit as original_thickness.
    """
    # Apply the area conservation assumption
    stretched_thickness = original_thickness / (stretch_factor ** 2)

    return stretched_thickness


if __name__ == "__main__":
    """ 
    NOTE:   
        Type of modulus     Symbol      Describes
        ------------------------------------------
    1.  Young's modulus     E           Axial stretching or compression
    2.  Shear modulus       G or mu     Response to shear (sliding layers)
    3.  Bulk modulus        K           Resistance to uniform compression (volume change)
    4.  Biaxial modulus     E/(1-mu^2)  In-plane stretching of membrane under plane stress
    """
    # Example Usage
    original_thickness = 20.0  # microns
    stretch_factor = 1.2  # 20% biaxial stretch

    new_thickness = calculate_stretched_thickness(original_thickness, stretch_factor)
    print(f"The new thickness of the membrane after stretching is {new_thickness:.2f} microns.")
