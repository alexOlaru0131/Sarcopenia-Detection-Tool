###### prepare_masks.py ######
# ->

###### IMPORTS ######
from imports import *
###### END IMPORTS ######

#
dirpath = os.path.join('dataset')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
input_folder = ["C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/Antal viorica regi CT/ANTAL VIORICA 20240326133207/01_CT AbdomenPelvis Nativ(Adult)/02_AbdomenPelvis 1,50 Br40 ax",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BALAZS JOLAN 20240410140331/01_CT AbdomenPelvis Nativ(Adult)/02_AbdomenPelvis 1,50 Br40 ax",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BALINT ZSOLT-LEVENTE 20240327120203",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BERECZKI FRANCISC 20240327124557/01_Private AbdMultiPhase (Adult)/02_NATIV  15  B31f",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BIRO IOSIF 20240408135652",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BOBRIC SIMION 20240326125731",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BORDI ANDREI 20240326131709",
                "C:/Sarcopenia Detection Tool/Baza de date CT sarcopenie/training/BOSCA LUCRETIA 20240318162649/01_Private AbdMultiPhase (Adult)/02_NATIV  15  B31f",]
output_image_folder = "dataset/images"
output_mask_folder = "dataset/masks"
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)
#

#
def generate_precise_mask(hu_image):
    bone_mask = (hu_image > 130).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bone_mask, connectivity=8)
    verteber_mask = np.zeros_like(bone_mask)
    area_mask = np.zeros_like(bone_mask)

    height, length = bone_mask.shape

    minimal_area = 50

    used = set()
    for i in range(1, num_labels):
        if i in used:
            continue
        ci = centroids[i]
        region = (labels == i).astype(np.uint8)
        group = region.copy()

        area = stats[i, cv2.CC_STAT_AREA]
        if area >= minimal_area:

            M = cv2.moments(region)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                if length // 2 - 150 < cx < length // 2 + 150 and \
                   height // 2 - 150 < cy < height // 2 + 150:
                    area_mask[labels == i] = 1
                    for j in range(i + 1, num_labels):
                        if j in used:
                            continue
                        cj = centroids[j]
                        dist = np.linalg.norm(ci - cj)

                        if dist < 200:
                            group = group | (labels == j).astype(np.uint8)
                            used.add(j)
        used.add(i)

        if cv2.countNonZero(group) > cv2.countNonZero(verteber_mask):
            verteber_mask = group

    kernel = np.ones((3, 3), np.uint8)
    verteber_mask = cv2.morphologyEx(verteber_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    verteber_mask = cv2.bitwise_and(verteber_mask, area_mask)

    return verteber_mask

#
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

#
photo_index_vector = []
photo_index = 0
photo_index_vector.append(photo_index)
for folder_string in tqdm(input_folder):
    for filename in os.listdir(folder_string):
        if filename.lower().strip().endswith(".dcm"):
            photo_index += 1
            dcm_path = os.path.join(folder_string, filename)
            generate_image_and_mask(dcm_path, photo_index)
    photo_index_vector.append(photo_index)

print(photo_index_vector)