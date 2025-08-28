import tkinter as tk

root = tk.Tk()
if tk._default_root is not None:
    try:
        root.destroy()
    except Exception:
        pass

import numpy as np
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from checkVerteber import test_model
from detectMuscles import detect_muscles
from analyseAreas import calculate_areas
from functools import partial
import pydicom
import os
import dicom2jpg

def guess_sex_from_name(name):
    if not name or name == "-":
        return "-"
        # Dă jos orice prefix DICOM: Nume^Prenume
    try:
        parts = str(name).replace(".", " ").split("^")
        prenume = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    except Exception:
        prenume = str(name)
    prenume = prenume.split()[0]  # Ia doar primul prenume
    if prenume.lower().endswith("a"):
        return "F"
    return "M"

def fig_to_pil_image(fig, dpi=100):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img

import re

def extract_dicom_info(ds):
    nume = getattr(ds, "PatientName", "-")
    sex = getattr(ds, "PatientSex", "-")
    if not sex or sex == "-" or sex == "O":
        sex = guess_sex_from_name(nume)

    # Extrage vârsta ca număr de ani (sau "-" dacă nu există)
    varsta_raw = getattr(ds, "PatientAge", "-")
    if varsta_raw and varsta_raw != "-":
        match = re.match(r"(\d+)", str(varsta_raw))
        if match:
            varsta_nr = int(match.group(1))
        else:
            varsta_nr = "-"
    else:
        varsta_nr = "-"

    info = {
        "Nume": nume,
        "ID": getattr(ds, "PatientID", "-"),
        "Sex": sex,
        "Vârstă": varsta_nr,  # <-- ca număr
        "Înălțime": getattr(ds, "PatientSize", "-"),
        "Greutate": getattr(ds, "PatientWeight", "-"),
        "Study date": getattr(ds, "StudyDate", "-"),
        "Institution": getattr(ds, "InstitutionName", "-"),
        "Modality": getattr(ds, "Modality", "-")
    }
    return info


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interfață Detecție Sarcopenie")
        self.geometry("1400x650")
        self.minsize(1200, 600)
        self.configure(bg='#f0f0f0')

        self.left_figs = [None, None, None]
        self.fig_final = None
        self.image_np = None
        self.mask_np = None
        self.center_is_detect_muscles = False
        self.params_by_imgidx = {}
        self.muscle_imgs_by_idx = {}
        self.saved_regions_by_idx = {}
        self.y_imgs_np = []

        self.dicom_path = None
        self.dicom_dataset = None
        self.dicom_info_labels = {}
        self.flip_vertical = False
        self.area_min_var = tk.DoubleVar(value=20)
        self.zones_visible = {
            "yellow": tk.BooleanVar(value=True),
            "green": tk.BooleanVar(value=True),
            "pink": tk.BooleanVar(value=True),
            "blue": tk.BooleanVar(value=True)
        }
        self.pixel_spacing_mm = 1.0
        self.saved_regions = {"yellow": [], "green": [], "pink": [], "blue": []}

        self.main_frame = tk.Frame(self, bg='#f0f0f0')
        self.main_frame.pack(fill='both', expand=True)
        self.main_frame.columnconfigure(0, weight=1, minsize=220)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.columnconfigure(2, weight=1, minsize=320)
        self.main_frame.rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(self.main_frame, bg='#d3d3d3')
        self.left_panel.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        # self._populate_left_panel()  # NU o chema aici!

        self.center_panel = tk.Frame(self.main_frame, bg='#e0e0e0')
        self.center_panel.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        # self._populate_center_panel()  # NU o chema aici!

        # ----------- SCROLLABLE RIGHT PANEL FIXED WIDTH -----------
        panel_width = 350  # sau orice lățime vrei tu (ex: 320)
        self.right_panel_canvas = tk.Canvas(self.main_frame, bg='#d3d3d3', highlightthickness=0, width=panel_width)
        self.right_panel_canvas.grid(row=0, column=2, sticky="nswe", padx=10, pady=10)
        self.right_panel_scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.right_panel_canvas.yview)
        self.right_panel_scrollbar.grid(row=0, column=3, sticky="ns", pady=10)
        self.right_panel = tk.Frame(self.right_panel_canvas, bg='#d3d3d3', width=panel_width)
        self.right_panel_window = self.right_panel_canvas.create_window((0, 0), window=self.right_panel, anchor="nw", width=panel_width)
        def on_right_panel_configure(event):
            # Obligă frame-ul să aibă fix lățimea panoului, nu după conținut!
            self.right_panel_canvas.itemconfig(self.right_panel_window, width=panel_width)
            self.right_panel_canvas.configure(scrollregion=self.right_panel_canvas.bbox("all"), width=panel_width)
        self.right_panel.bind("<Configure>", on_right_panel_configure)
        self.right_panel_canvas.configure(yscrollcommand=self.right_panel_scrollbar.set)
        self.right_panel_canvas.bind_all("<MouseWheel>", lambda event: self.right_panel_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        # ------------------------------------------------

        self.lower_var = tk.DoubleVar(value=33)
        self.upper_var = tk.DoubleVar(value=100)
        self.max_dist_var = tk.DoubleVar(value=190)
        self.horiz_margin_var = tk.DoubleVar(value=130)
        self.cross_var = tk.DoubleVar(value=30)
        self.right_panel_controls = tk.Frame(self.right_panel, bg='#d3d3d3', width=panel_width)
        self._populate_right_panel()
        
        # ---- FRAME DOAR PENTRU ÎNCEPUT (buton DICOM) ----
        self.right_panel_init = tk.Frame(self.right_panel, bg='#d3d3d3', width=panel_width)
        self.right_panel_init.pack(fill='both', expand=True)
        self.btn_load_dicom_init = tk.Button(
            self.right_panel_init, text="Încarcă DICOM", command=self.on_load_dicom, width=int((panel_width-30)//9)
        )
        self.btn_load_dicom_init.pack(expand=True, pady=30)
        
        
        self.bind("<Configure>", self._resize_center_image)
        self.center_fig = None
        self.center_fig = self.fig_final
        
    def _on_left_click(self, event, idx):
        self.swap_with_center(idx)
        
    def swap_with_center(self, idx):
        # Salvezi temporar centrul
        center_img = self.center_img_orig
        center_type = self.center_img_type
        center_idx = getattr(self, "current_img_idx", 0)
    
        # Slotul selectat din stânga
        slot = self.left_slots[idx]
        # SWAP img/type/idx între centru și slotul selectat
        self.left_slots[idx] = {
            "img": center_img,
            "type": center_type,
            "idx": center_idx,
        }
        self.center_img_orig = slot["img"]
        self.center_img_type = slot["type"]
        self.current_img_idx = slot["idx"]
    
        # Update stânga
        for i, slot in enumerate(self.left_slots):
            img = slot["img"]
            tk_img = ImageTk.PhotoImage(img.resize((200, 180), resample=Image.LANCZOS))
            self.left_labels[i].config(image=tk_img, text=slot["type"])
            self.left_labels[i].image = tk_img
    
        # Update centru
        w = self.center_panel.winfo_width()
        h = self.center_panel.winfo_height()
        try:
            RESAMPLE = Image.Resampling.LANCZOS
        except AttributeError:
            RESAMPLE = Image.LANCZOS
        img_center_resized = self.center_img_orig.resize((max(w-40, 1), max(h-40, 1)), resample=RESAMPLE)
        self.center_tkimg = ImageTk.PhotoImage(img_center_resized)
        self.center_label.config(image=self.center_tkimg)
        self.center_label.image = self.center_tkimg
    
        self.center_is_detect_muscles = (self.center_img_type == "detect_muscles")
        self.update_sliders_visibility()
        self.load_params_for_idx(self.current_img_idx)
        self.update_center_img_from_params()

    def _populate_left_panel(self):
        self.left_labels = []
        for i, slot in enumerate(self.left_slots):
            img = slot["img"]
            tk_img = ImageTk.PhotoImage(img.resize((200, 180), resample=Image.LANCZOS))
            label = tk.Label(self.left_panel, image=tk_img, bg='#d3d3d3', cursor="hand2")
            label.image = tk_img
            label.pack(pady=15, fill='x', expand=True)
            self.left_labels.append(label)
            label.bind("<Button-1>", partial(self._on_left_click, idx=i))

    def _populate_center_panel(self):
        # Inițializare imagine centrală cu masca musculară default
        filtered, saved_regions = detect_muscles(
            self.image_np,
            self.mask_np,
            max_distance=190,
            horizontal_margin=130,
            cross_threshold=30,
            lower_threshold=33,
            upper_threshold=100
        )
        img_show = (filtered * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_show).convert("L")
        img_pil = img_pil.convert("RGB")
        self.center_img_orig = img_pil
        self.center_img = self.center_img_orig.copy()
        self.center_tkimg = ImageTk.PhotoImage(self.center_img)
        self.center_label = tk.Label(self.center_panel, image=self.center_tkimg, bg='#e0e0e0')
        self.center_label.pack(expand=True, fill='both')
        # Salvează regiuni pentru arii inițiale (default spacing)
        self.saved_regions = saved_regions

    def set_sliders_visible(self, visible: bool):
        # Sliderele și zona cu arii/textul apar doar dacă visible==True (adică dacă e masca musculară în centru)
        for lbl in self.slider_labels:
            if visible:
                lbl.pack()
            else:
                lbl.pack_forget()
        for slider in self.sliders:
            if visible:
                slider.pack(fill='x', pady=8, padx=15)
            else:
                slider.pack_forget()
        if visible:
            self.area_min_slider.pack(fill='x', pady=8, padx=15)
        else:
            self.area_min_slider.pack_forget()
        for key in self.area_labels:
            parent = self.area_labels[key].master
            if visible:
                parent.pack(anchor="w", padx=15, pady=2)
            else:
                parent.pack_forget()
        # Textul cu info DICOM apare doar când visible==True (doar pe masca musculară)
        if visible:
            self.info_frame.pack(fill='x', pady=(25, 10), padx=10)
        else:
            self.info_frame.pack_forget()

    def _populate_right_panel(self):
        panel_width = 350
        parent = self.right_panel_controls
        self.slider_labels = []
    
        # Header
        lbl_header = tk.Label(
            parent,
            text="Parametri modificabili",
            font=('Arial', 18),
            bg='#d3d3d3',
            wraplength=panel_width-20,
            justify='center',
            width=int((panel_width-30)//9)
        )
        lbl_header.pack(pady=10, padx=8)
        self.slider_labels.append(lbl_header)
    
        # Sliders
        slider_defs = [
            ("Prag jos - valori pixeli mușchi", self.lower_var, 0, 150),
            ("Prag sus - valori pixeli mușchi", self.upper_var, 50, 255),
            ("Distanța față de centrul vertebrei", self.max_dist_var, 10, 400),
            ("Distanța pe orizontală față de centrul vertebrei", self.horiz_margin_var, 0, 300),
            ("Grosimea liniei pe centrul vertebrei", self.cross_var, 0, 100),
        ]
        self.sliders = []
    
        for label_text, var, vmin, vmax in slider_defs:
            lbl = tk.Label(
                parent,
                text=label_text,
                bg='#d3d3d3',
                wraplength=panel_width-25,
                anchor='w',
                width=int((panel_width-30)//9)
            )
            lbl.pack(padx=8, anchor='w')
            self.slider_labels.append(lbl)
            slider = ttk.Scale(
                parent, from_=vmin, to=vmax, orient='horizontal', variable=var,
                command=lambda val, v=var: self.on_slider_change()
            )
            slider.pack(fill='x', pady=8, padx=15)
            self.sliders.append(slider)
    
        # Suprafață minimă
        lbl_area = tk.Label(
            parent, text="Suprafață minimă pentru mușchi", bg='#d3d3d3',
            wraplength=panel_width-25, anchor='w', width=int((panel_width-30)//9)
        )
        lbl_area.pack(pady=(18,2), padx=8, anchor='w')
        self.slider_labels.append(lbl_area)
        slider_area = ttk.Scale(parent, from_=1, to=1000, orient='horizontal', variable=self.area_min_var,
                                command=lambda val: self.on_slider_change())
        slider_area.pack(fill='x', pady=8, padx=15)
        self.area_min_slider = slider_area
    
        # Zone vizibile
        lbl_zones = tk.Label(
            parent, text="Zone vizibile:", font=('Arial', 12), bg='#d3d3d3',
            wraplength=panel_width-25, anchor='w', width=int((panel_width-30)//9)
        )
        lbl_zones.pack(pady=(18, 2), padx=8, anchor='w')
        self.slider_labels.append(lbl_zones)
        # ... restul la fel ...

        color_labels = [
            ("Galben", "yellow"),
            ("Verde", "green"),
            ("Roz", "pink"),
            ("Albastru", "blue")
        ]
        self.area_labels = {}
    
        for text, key in color_labels:
            frame = tk.Frame(parent, bg='#d3d3d3', width=panel_width-32)
            frame.pack(anchor="w", padx=15, pady=2)
            chk = tk.Checkbutton(
                frame, text=text, variable=self.zones_visible[key], bg='#d3d3d3',
                command=self.on_slider_change, width=10
            )
            chk.pack(side="left")
            area_label = tk.Label(frame, text="Arie: -- cm²", bg='#d3d3d3', width=16)
            area_label.pack(side="left", padx=(10,0))
            self.area_labels[key] = area_label
    
        # --- Buton FLIP imagine ---
        self.btn_flip = tk.Button(parent, text="Întoarce imaginea pe verticală", command=self.on_flip_vertical, width=int((panel_width-30)//9))
        self.btn_flip.pack(pady=16, padx=8)
    
        # ---- Buton pentru selectare DICOM ----
        btn = tk.Button(parent, text="Încarcă DICOM", command=self.on_load_dicom, width=int((panel_width-30)//9))
        btn.pack(pady=10, padx=8)
    
        # ---- Zona info DICOM ----
        self.info_frame = tk.Frame(parent, bg='#d3d3d3', width=panel_width-25)
        self.info_frame.pack(fill='x', pady=(25, 10), padx=10)
        self._update_dicom_info(None)  # Initializează cu "-"
    
        # La final, afișează ariile inițiale
        self.update_area_labels()

    def _update_dicom_info(self, ds):
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        if ds is None:
            info = {
                "Nume": "-",
                "ID": "-",
                "Sex": "-",
                "Vârstă": "-",
                "Înălțime": "-",
                "Greutate": "-",
                "Study date": "-",
                "Institution": "-",
                "Modality": "-"
            }
        else:
            info = extract_dicom_info(ds)
        self.dicom_info_labels = {}
        for key, val in info.items():
            row = tk.Frame(self.info_frame, bg='#d3d3d3')
            row.pack(fill='x')
            lbl1 = tk.Label(row, text=f"{key}:", anchor='w', bg='#d3d3d3', width=10)
            lbl1.pack(side='left')
            lbl2 = tk.Label(row, text=str(val), anchor='w', bg='#d3d3d3')
            lbl2.pack(side='left')
            self.dicom_info_labels[key] = lbl2

    def load_params_for_idx(self, idx):
        params = self.params_by_imgidx.get(idx, {})
        self.lower_var.set(params.get("lower", 33))
        self.upper_var.set(params.get("upper", 100))
        self.max_dist_var.set(params.get("max_dist", 190))
        self.horiz_margin_var.set(params.get("horiz_margin", 130))
        self.cross_var.set(params.get("cross", 30))
        self.area_min_var.set(params.get("area_min", 20))
        for k in self.zones_visible:
            self.zones_visible[k].set(params.get("zones", {}).get(k, True))

    def apply_window(self, image, center, width):
        """Aplică windowing la imagine."""
        img_min = center - width // 3
        img_max = center + width // 3
        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min) * 255.0
        return image.astype(np.uint8)

    def on_load_dicom(self):
        folder_path = filedialog.askdirectory(title="Selectează folderul cu imagini DICOM")
        if folder_path:
            dicom_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(".dcm"):
                    dicom_files.append(os.path.join(folder_path, file))
            dicom_files.sort()
            if not dicom_files:
                messagebox.showerror("Eroare", "Nu s-au găsit fișiere DICOM (.dcm) în acest folder.")
                return
    
            if dicom_files:
                ds_first = pydicom.dcmread(dicom_files[0])
                self._update_dicom_info(ds_first)
                if hasattr(ds_first, "PixelSpacing"):
                    spacing = ds_first.PixelSpacing
                    if isinstance(spacing, pydicom.multival.MultiValue):
                        row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
                    else:
                        row_spacing = col_spacing = float(spacing)
                    self.pixel_spacing_mm = (row_spacing + col_spacing) / 2
                else:
                    self.pixel_spacing_mm = 1.0
            else:
                self._update_dicom_info(None)
                self._update_dicom_info(None)
    
            image_list = []
            pixel_spacings = []
            for dicom_file in dicom_files:
                ds = pydicom.dcmread(dicom_file)
                image = ds.pixel_array.astype(np.float32)
                intercept = float(ds.get("RescaleIntercept", 0.0))
                slope = float(ds.get("RescaleSlope", 1.0))
                image = image * slope + intercept
                center = ds.get("WindowCenter", None)
                width = ds.get("WindowWidth", None)
                if isinstance(center, pydicom.multival.MultiValue): center = center[0]
                if isinstance(width, pydicom.multival.MultiValue): width = width[0]
                image_list.append(image)
                try:
                    center = float(center)
                    width = float(width)
                except Exception:
                    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
                    image = image.astype(np.uint8)
                image_list.append(image)
                
            # Ascunde frame-ul inițial cu buton
            self.right_panel_init.pack_forget()
            # Afișează frame-ul cu controalele
            self.right_panel_controls.pack(fill='both', expand=True)
    
            # ====== test_model ======
            y_fig1, y_fig2, y_fig3, fig_final, img_np_final, pred_mask_main_final, center_y_final, center_x_final, y_np1, y_np2, y_np3, mask1, mask2, mask3 = test_model(image_list)
            self.left_slots = [
                {"img": fig_to_pil_image(y_fig1), "type": "Y1", "idx": 1},
                {"img": fig_to_pil_image(y_fig2), "type": "Y2", "idx": 2},
                {"img": fig_to_pil_image(y_fig3), "type": "Y3", "idx": 3},
            ]
            
            self.y_imgs_np = [img_np_final, y_np1, y_np2, y_np3]
            self.y_masks_np = [pred_mask_main_final, mask1, mask2, mask3]
            self.left_figs = [y_fig1, y_fig2, y_fig3]
            self.left_types = ['Y1', 'Y2', 'Y3']
            self.fig_final = fig_final
            self.image_np = img_np_final
            self.mask_np = pred_mask_main_final
            self.detect_muscles_idx = 3
    
            # ====== detect_muscles (cu sliderii deja inițializați) ======
            filtered, saved_regions = detect_muscles(
                self.image_np,
                self.mask_np,
                max_distance=self.max_dist_var.get(),
                horizontal_margin=self.horiz_margin_var.get(),
                cross_threshold=self.cross_var.get(),
                lower_threshold=self.lower_var.get(),
                upper_threshold=self.upper_var.get(),
                min_area=self.area_min_var.get(),
                show_zones={k: v.get() for k, v in self.zones_visible.items()},
                pred_mask_main=self.mask_np
            )
            self.saved_regions = saved_regions
            img_show = (filtered * 255).astype(np.uint8) if filtered.max() <= 1.1 else filtered.astype(np.uint8)
            img_pil = Image.fromarray(img_show).convert("L")
            img_pil = img_pil.convert("RGB")
            self.center_img_orig = img_pil
    
            # === update UI ===
            self._populate_left_panel()
            w = self.center_panel.winfo_width()
            h = self.center_panel.winfo_height()
            try:
                RESAMPLE = Image.Resampling.LANCZOS
            except AttributeError:
                RESAMPLE = Image.LANCZOS
            img_center_resized = img_pil.resize((max(w-40,1), max(h-40,1)), resample=RESAMPLE)
            self.center_tkimg = ImageTk.PhotoImage(img_center_resized)
            if not hasattr(self, "center_label") or self.center_label is None:
                self.center_label = tk.Label(self.center_panel, image=self.center_tkimg, bg='#e0e0e0')
                self.center_label.pack(expand=True, fill='both')
            else:
                self.center_label.config(image=self.center_tkimg)
            self.center_label.image = self.center_tkimg
    
            self.update_area_labels()
            messagebox.showinfo("Succes", f"A fost încărcat folderul și procesat primul Y!")
            self.center_img_type = "detect_muscles"
            self.center_is_detect_muscles = True
            self.update_sliders_visibility()
            
            if pixel_spacings:
                row_spacing, col_spacing = pixel_spacings[0]  # sau calculezi media, dacă e nevoie
                self.pixel_spacing_mm = (row_spacing + col_spacing) / 2  # media pentru arii, ca float
            else:
                self.pixel_spacing_mm = 1.0
                
            self.params_by_imgidx = {}
            self.muscle_imgs_by_idx = {}
            self.saved_regions_by_idx = {}
            
            imgs = [img_np_final] + [np.array(fig_to_pil_image(fig_final), dtype=np.float32) for fig_final in [y_fig1, y_fig2, y_fig3]]
            masks = [pred_mask_main_final] * 4  # Sau dacă ai altă mască pentru Y-uri, modifică aici
            
            # Parametrii default (poți să-i schimbi după preferință)
            default_params = {
                "lower": self.lower_var.get(),
                "upper": self.upper_var.get(),
                "max_dist": self.max_dist_var.get(),
                "horiz_margin": self.horiz_margin_var.get(),
                "cross": self.cross_var.get(),
                "area_min": self.area_min_var.get(),
                "zones": {k: v.get() for k, v in self.zones_visible.items()}
            }
            imgs = self.y_imgs_np
            masks = self.y_masks_np
            for idx, (img, mask) in enumerate(zip(imgs, masks)):
                self.params_by_imgidx[idx] = default_params.copy()
                filtered, saved_regions = detect_muscles(
                    img,
                    mask,
                    max_distance=default_params["max_dist"],
                    horizontal_margin=default_params["horiz_margin"],
                    cross_threshold=default_params["cross"],
                    lower_threshold=default_params["lower"],
                    upper_threshold=default_params["upper"],
                    min_area=default_params["area_min"],
                    show_zones=default_params["zones"],
                    pred_mask_main=mask
                )
                self.muscle_imgs_by_idx[idx] = filtered
                self.saved_regions_by_idx[idx] = saved_regions

    def update_center_img_from_params(self):
        idx = getattr(self, "current_img_idx", 0)
        filtered = self.muscle_imgs_by_idx.get(idx, None)
        if filtered is None:
            return
        img_pil = Image.fromarray((filtered * 255).astype(np.uint8) if filtered.max() <= 1.1 else filtered.astype(np.uint8))
        img_pil = img_pil.resize(self.center_img_orig.size)
        if self.flip_vertical:
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        self.center_img_orig = img_pil
        w = self.center_panel.winfo_width()
        h = self.center_panel.winfo_height()
        try:
            RESAMPLE = Image.Resampling.LANCZOS
        except AttributeError:
            RESAMPLE = Image.LANCZOS
        img_center_resized = img_pil.resize((max(w-40, 1), max(h-40, 1)), resample=RESAMPLE)
        self.center_tkimg = ImageTk.PhotoImage(img_center_resized)
        self.center_label.config(image=self.center_tkimg)
        self.center_label.image = self.center_tkimg
        self.saved_regions = self.saved_regions_by_idx[idx]
        self.update_area_labels()

    def on_flip_vertical(self):
        self.flip_vertical = not self.flip_vertical
    
        # Folosești self.center_img_orig ca original
        img = self.center_img_orig
        if self.flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        w = self.center_panel.winfo_width()
        h = self.center_panel.winfo_height()
        try:
            RESAMPLE = Image.Resampling.LANCZOS
        except AttributeError:
            RESAMPLE = Image.LANCZOS
        img_center_resized = img.resize((max(w-40, 1), max(h-40, 1)), resample=RESAMPLE)
        self.center_tkimg = ImageTk.PhotoImage(img_center_resized)
        self.center_label.config(image=self.center_tkimg)
        self.center_label.image = self.center_tkimg

    def on_slider_change(self, *args):
        idx = getattr(self, "current_img_idx", 0)
        params = {
            "lower": self.lower_var.get(),
            "upper": self.upper_var.get(),
            "max_dist": self.max_dist_var.get(),
            "horiz_margin": self.horiz_margin_var.get(),
            "cross": self.cross_var.get(),
            "area_min": self.area_min_var.get(),
            "zones": {k: v.get() for k, v in self.zones_visible.items()}
        }
        self.params_by_imgidx[idx] = params
        # Reconstruiește imaginea pentru acești parametri:
        img = self.y_imgs_np[idx]
        mask = self.y_masks_np[idx]  # (dacă fiecare Y are aceeași mască, altfel salvezi și măștile separat)
        filtered, saved_regions = detect_muscles(
            img,
            mask,
            max_distance=params["max_dist"],
            horizontal_margin=params["horiz_margin"],
            cross_threshold=params["cross"],
            lower_threshold=params["lower"],
            upper_threshold=params["upper"],
            min_area=params["area_min"],
            show_zones=params["zones"],
            pred_mask_main=mask
        )
        self.muscle_imgs_by_idx[idx] = filtered
        self.saved_regions_by_idx[idx] = saved_regions
        self.update_center_img_from_params()

    def _resize_center_image(self, event):
        if not hasattr(self, "center_img_orig") or self.center_img_orig is None:
            return  # NU face resize dacă nu există imaginea!
        w = self.center_panel.winfo_width()
        h = self.center_panel.winfo_height()
        if w > 10 and h > 10:
            try:
                RESAMPLE = Image.Resampling.LANCZOS
            except AttributeError:
                RESAMPLE = Image.LANCZOS
            img = self.center_img_orig.resize((w-40, h-40), resample=RESAMPLE)
            self.center_tkimg = ImageTk.PhotoImage(img)
            self.center_label.config(image=self.center_tkimg)
            self.center_label.image = self.center_tkimg


    def update_area_labels(self):
        if hasattr(self, "saved_regions") and hasattr(self, "pixel_spacing_mm"):
            try:
                areas, diagnostic = calculate_areas(self.saved_regions, self.pixel_spacing_mm)
                for key in self.area_labels:
                    area_val = areas.get(key, 0.0)
                    self.area_labels[key].config(text=f"Arie: {area_val:.2f} cm²")
            except Exception:
                for key in self.area_labels:
                    self.area_labels[key].config(text="Arie: -- cm²")
        else:
            for key in self.area_labels:
                self.area_labels[key].config(text="Arie: -- cm²")
                
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
