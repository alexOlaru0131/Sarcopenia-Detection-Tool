import os
import pydicom
import numpy as np
import cv2

input_folder = "P2.1_3. Baza de date CT sarcopenie/Antal Viorica/ANTAL VIORICA 20240326133207/01_CT AbdomenPelvis Nativ(Adult)/02_AbdomenPelvis 1,50 Br40 ax"
output_image_folder = "dataset/images"
output_mask_folder = "dataset/masks"
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

def generate_precise_mask(hu_image):
    bone_mask = (hu_image > 100).astype(np.uint8)

    h, w = bone_mask.shape
    spine_mask = np.zeros_like(bone_mask)
    spine_mask[int(h*0.2):int(h*0.9), int(w*0.2):int(w*0.9)] = 1
    bone_mask = bone_mask * spine_mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bone_mask, connectivity=8)

    vertebra_mask = np.zeros_like(bone_mask)
    used = set()
    for i in range(1, num_labels):
        if i in used:
            continue
        ci = centroids[i]
        region_mask = (labels == i).astype(np.uint8)
        group = region_mask.copy()

        for j in range(i + 1, num_labels):
            if j in used:
                continue
            cj = centroids[j]
            dist = np.linalg.norm(ci - cj)
            if dist < 100:
                group = group | (labels == j).astype(np.uint8)
                used.add(j)
        used.add(i)

        if cv2.countNonZero(group) > cv2.countNonZero(vertebra_mask):
            vertebra_mask = group

    kernel = np.ones((3, 3), np.uint8)
    vertebra_mask = cv2.morphologyEx(vertebra_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return vertebra_mask

def generate_image_and_mask(dcm_path, idx):
    ds = pydicom.dcmread(dcm_path)
    image = ds.pixel_array.astype(np.float32)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    slope = getattr(ds, 'RescaleSlope', 1)
    hu = image * slope + intercept
    bone_mask = generate_precise_mask(hu)

    img_name = f"img_{idx:04d}.png"
    mask_name = f"mask_{idx:04d}.png"
    cv2.imwrite(os.path.join(output_image_folder, img_name), hu)
    cv2.imwrite(os.path.join(output_mask_folder, mask_name), bone_mask * 255)

for i, file in enumerate(os.listdir(input_folder)):
    if file.lower().endswith(".dcm"):
        generate_image_and_mask(os.path.join(input_folder, file), i)

print("Măștile au fost generate!")
