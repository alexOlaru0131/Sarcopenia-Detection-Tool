###### analyse_muscles.py ######
# ->

###### IMPORTS ######
from imports import *
###### END IMPORTS ######

def detect_muscles(
        image_np, mask_np, 
        line_direction=None, line_point=None,
        max_distance=190, horizontal_margin=130, cross_threshold=30,
        lower_threshold=33, upper_threshold=100, pred_mask_main=None,
        min_area=20, show_zones=None):

    if show_zones is None:
        show_zones = {"yellow": True, "green": True, "pink": True, "blue": True}

    saved_regions = {"yellow": [], "green": [], "pink": [], "blue": []}
    vertebra_mask = (mask_np > 0)
    vertebra_dilated = binary_dilation(vertebra_mask, iterations=1)
    image = gaussian_filter(image_np.copy(), sigma=0.1)
    image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)

    muscle_mask = (image > lower_threshold) & (image < upper_threshold)
    muscle_mask = binary_fill_holes(muscle_mask) 

    labeled = sk_label(muscle_mask)
    filtered = np.zeros_like(muscle_mask, dtype=np.uint8) 

    bounding_box = find_objects(mask_np.astype(int))
    if bounding_box:
        top_edge = bounding_box[0][0].start
    else:
        top_edge = int(center_of_mass(mask_np)[0])
    center_y, center_x = center_of_mass(mask_np)
    vertebra_center = (top_edge + 10, center_x)

    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        cy, cx = region.centroid
        dy, dx = cy - vertebra_center[0], cx - vertebra_center[1]
        distance = np.sqrt(dy**2 + dx**2)
        if cy > center_y:
            horizontal_check = True
        else:
            horizontal_check = abs(cx - vertebra_center[1]) < horizontal_margin
        if (distance < max_distance and
            cy > vertebra_center[0] and
            horizontal_check and
            not np.any((labeled == region.label) & vertebra_mask) and
            not np.any((labeled == region.label) & vertebra_dilated)):
            filtered[labeled == region.label] = 1

    if line_direction is not None and line_point is not None:
        yy, xx = np.meshgrid(np.arange(filtered.shape[0]), np.arange(filtered.shape[1]), indexing='ij')
        line_vec = np.array(line_direction)
        point_vec = np.stack([xx - line_point[0], yy - line_point[1]], axis=-1)
        cross = np.abs(line_vec[1] * point_vec[..., 0] - line_vec[0] * point_vec[..., 1]) / np.linalg.norm(line_vec)
        close_to_line = (cross < cross_threshold)
        filtered[close_to_line] = 0
        
    labeled_muscle = sk_label(filtered)

    background_rgb = np.stack([image]*3, axis=-1).astype(np.float32) / 255.0

    alpha_mask = 0.45
    out_rgb = background_rgb.copy()

    for region in regionprops(labeled_muscle):
        cy, cx = region.centroid
        side_mask = (labeled_muscle == region.label)
        zone_name = None
        if cx < center_x and region.bbox[0] < center_y:
            zone_name = "yellow"
            color = [1.0, 1.0, 0.0]
        elif cx >= center_x and region.bbox[0] < center_y:
            zone_name = "green"
            color = [0.0, 1.0, 0.0]
        elif cx >= center_x and region.bbox[0] > center_y:
            zone_name = "pink"
            color = [1.0, 0.0, 1.0]
        elif cx < center_x and region.bbox[0] > center_y:
            zone_name = "blue"
            color = [0.0, 0.0, 1.0]
        else:
            continue
    
        if not show_zones.get(zone_name, True):
            continue
    
        # --- ADAUGĂ MASCA ÎN DICT ---
        saved_regions[zone_name].append(side_mask.astype(np.uint8))
    
        for c in range(3):
            out_rgb[..., c][side_mask] = (
                alpha_mask * color[c] + (1 - alpha_mask) * out_rgb[..., c][side_mask]
            )


    if pred_mask_main is not None:
        vertebra_mask = (pred_mask_main > 0)
        for c in range(3):
            overlay_color = [1.0, 0.0, 0.0]
            out_rgb[..., c][vertebra_mask] = (
                0.35 * overlay_color[c] + 0.65 * out_rgb[..., c][vertebra_mask]
            )

    out_rgb_uint8 = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)

    return out_rgb_uint8, saved_regions
