###### calcule_muscle_area.py ######
# ->

###### IMPORTS ######
from imports import *
###### END IMPORTS ######

def calculate_areas(saved_regions, pixel_spacing_mm):
    # pixel_spacing_mm trebuie să fie (row_spacing, col_spacing)
    if not isinstance(saved_regions, dict):
        # Early fail-safe
        print("ERROR: saved_regions nu este dict, ci", type(saved_regions))
        return {}, ""
    if isinstance(pixel_spacing_mm, (int, float)):
        # fallback, dacă e scalar
        row_spacing = col_spacing = pixel_spacing_mm
    else:
        row_spacing, col_spacing = pixel_spacing_mm

    pixel_area_cm2 = (row_spacing * col_spacing) / 100
    area_dict = {}
    diagnostic = 'Sănătos'

    for color, regions in saved_regions.items():
        total_pixel_count = sum(np.sum(region_mask) for region_mask in regions)
        total_area_cm2 = total_pixel_count * pixel_area_cm2
        area_dict[color] = total_area_cm2

        if color in ('yellow', 'green') and total_area_cm2 < 6:
            diagnostic = f"Sarcopenie la mușchiul {color} având {total_area_cm2:.2f} cm^2."
    
    return area_dict, diagnostic
