import torch
from torchvision import transforms
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, center_of_mass, binary_dilation, binary_opening, binary_erosion
from scipy.signal import convolve2d
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import skeletonize
from UNet import UNet
import pydicom
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def count_branch_points(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbor_count = convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    return np.logical_and(skeleton, neighbor_count >= 3).sum()


def count_end_points(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbor_count = convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    return np.sum(np.logical_and(skeleton, neighbor_count == 1))


def is_y_shape(skeleton):
    skeleton_length = np.sum(skeleton)
    branches = count_branch_points(skeleton)
    end_points = count_end_points(skeleton)

    valid_length = skeleton_length < 250
    valid_branches = 1 <= branches <= 12
    valid_endpoints = 1 <= end_points <= 12

    return valid_length and valid_branches and valid_endpoints

def test_model(image_list):
    model_path_main = "vertebra_unet.pth"
    device = torch.device("cuda")

    model_main = UNet()
    model_main.load_state_dict(torch.load(model_path_main, map_location=device))
    model_main.to(device)
    model_main.eval()

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    previous_area, vertebra_index, y_count, last_y_frame_index = None, 0, 0, -10

    y_fig1, y_fig2, y_fig3 = None, None, None
    y_np1, y_np2, y_np3 = None, None, None  # <-- Adaugă astea!
    mask1, mask2, mask3 = None, None, None
    img_np_final = None
    pred_mask_main_final = None
    center_y_final, center_x_final = None, None
    extra_frames = 0

    for idx, img_np in enumerate(image_list):

        img_pil = Image.fromarray(img_np).convert("L")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output_main = model_main(img_tensor)
            pred_mask_main = (output_main > 0.8).float().squeeze().cpu().numpy()

        labeled_mask, num_labels = label(pred_mask_main)
        if num_labels == 0:
            continue

        areas = sorted([(i + 1, np.sum(labeled_mask == i + 1)) for i in range(num_labels)], key=lambda x: x[1], reverse=True)
        label_idx, area = areas[0]

        if previous_area is None:
            vertebra_index = 1
        elif abs(area - previous_area) / max(previous_area, 1e-5) > 0.1:
            vertebra_index += 1

        previous_area = area

        if not (vertebra_index >= 10):
            continue

        img_np = np.array(img_pil)
        expanded_mask = pred_mask_main.copy()
        dilated_mask = binary_dilation(expanded_mask, iterations=2)
        edge = dilated_mask & (~expanded_mask.astype(bool))
        expanded_mask[edge] = img_np[edge] > 150

        if idx - last_y_frame_index < 20:
            continue

        lower_half_mask = np.zeros_like(expanded_mask)
        lower_half_mask[expanded_mask.shape[0] // 2:, :] = expanded_mask[expanded_mask.shape[0] // 2:, :]

        binary_mask = (lower_half_mask > 0).astype(np.uint8)
        binary_mask = binary_opening(binary_mask, structure=np.ones((3, 3)))
        binary_mask = binary_erosion(binary_mask, iterations=1)

        if y_count == 3:
            extra_frames += 1
            if extra_frames == 20:
                fig_final, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img_np, cmap='gray')
    
                center_y, center_x = map(int, center_of_mass(pred_mask_main))
                img_np_final = img_np.copy()
                pred_mask_main_final = pred_mask_main.copy()
                center_y_final, center_x_final = center_y, center_x
                plt.close(fig_final)
                break
            else:
                continue

        skeleton = skeletonize(binary_mask)

        if not is_y_shape(skeleton):
            continue
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np, cmap='gray')
        skeleton_rgb = np.zeros((*skeleton.shape, 3), dtype=np.float32)
        skeleton_rgb[skeleton > 0] = [1.0, 0.0, 0.0]
        ax.imshow(skeleton_rgb, alpha=0.5)
        plt.axis('off')
        plt.close(fig)

        if y_count == 0:
            y_fig1 = fig
            y_np1 = img_np.copy()
            mask1 = pred_mask_main.copy()
        elif y_count == 1:
            y_fig2 = fig
            y_np2 = img_np.copy()
            mask2 = pred_mask_main.copy()
        elif y_count == 2:
            y_fig3 = fig
            y_np3 = img_np.copy()
            last_y_frame_index = idx
            mask3 = pred_mask_main.copy()

        y_count += 1
        last_y_frame_index = idx

    if y_fig1 is None or y_fig2 is None or y_fig3 is None or fig_final is None:
        raise RuntimeError("Nu s-au detectat 3 vertebre Y! Folderul DICOM nu conține date corecte sau modelul nu găsește vertebre.")
    return y_fig1, y_fig2, y_fig3, fig_final, img_np_final, pred_mask_main_final, center_y_final, center_x_final, y_np1, y_np2, y_np3, mask1, mask2, mask3
