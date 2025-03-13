#%% Imports -------------------------------------------------------------------

import nd2
import time
import shutil
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# functions
from functions import format_stack, merge_stack

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.filters.rank import median
from skimage.filters import threshold_otsu
from skimage.morphology import ball, h_maxima

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QGroupBox, QVBoxLayout, 
    )

#%% Comments ------------------------------------------------------------------

'''
- C1 : NRP2-EGFP (protein of interest)
- C2 : AF594-labelled virus
- C3 : EEA1 (early endosome marker)
- C4 : Hoechst (nucleus)
'''

#%% Inputs --------------------------------------------------------------------

# Procedure
overwrite = {
    "preprocess" : 1,
    "process" : 1,
    }

# Parameters
rf = 0.5
cyt_coeff = 0.5
ncl_coeff = 1.0

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

def save(stk, path, voxsize):
    if stk.ndim == 3: axes = "ZYX"
    if stk.ndim == 4: axes = "ZCYX"
    resolution = (1 / voxsize, 1 / voxsize)
    metadata = {"axes" : axes, "spacing" : voxsize, "unit" : "um"}
    tifffile.imwrite(
        path, stk, imagej=True, resolution=resolution, metadata=metadata)

#%% Function : preprocess() ---------------------------------------------------

def preprocess(path, rf=0.5):
        
    # Format stack
    stk, voxsize = format_stack(path, rf=rf)
        
    # Setup directory 
    dir_path = Path(data_path / path.stem)
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    dir_path.mkdir(exist_ok=True)
    
    # Save       
    stk_path = dir_path / (path.stem + f"_{voxsize}_stk.tif")
    save(stk, stk_path, voxsize)

#%% Function : process() ------------------------------------------------------

def process(paths, cyt_coeff=1, ncl_coeff=1):
    
    global cyts, ncls, cyts_med, ncls_med, cyts_msk, ncls_msk, cyts_thresh, ncls_thresh
    
    def _process(i, cyt, ncl):
        
        # Median filter
        cyt_med = median(cyt.copy(), footprint=ball(5))
        ncl_med = median(ncl.copy(), footprint=ball(5))
                
        # Get masks
        cyt_msk = cyt_med > (cyts_thresh * cyt_coeff)
        ncl_msk = ncl_med > (ncls_thresh * ncl_coeff)
        # cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
        # ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
        
        # Save
        dir_path = Path(data_path / paths[i].stem)
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        cyt_med_path = str(stk_path).replace("stk", "cyt_med")
        ncl_med_path = str(stk_path).replace("stk", "ncl_med")
        cyt_msk_path = str(stk_path).replace("stk", "cyt_msk")
        ncl_msk_path = str(stk_path).replace("stk", "ncl_msk")
        voxsize = float(str(stk_path.stem).split("_")[1])
        save(cyt_med, cyt_med_path, voxsize)
        save(ncl_med, ncl_med_path, voxsize)
        save(cyt_msk.astype("uint8"), cyt_msk_path, voxsize)
        save(ncl_msk.astype("uint8"), ncl_msk_path, voxsize)
        
        return cyt_med, ncl_med, cyt_msk, ncl_msk
        
    # Load data
    cyts, ncls = [], []
    for path in paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        stk = io.imread(stk_path)
        cyts.append(stk[..., 0]) # NRP2
        ncls.append(stk[..., 3]) # nuclei
        
    # Normalize
    cyts = norm_pct(cyts, sample_fraction=0.01)
    ncls = norm_pct(ncls, sample_fraction=0.01)
    cyts = [(cyt * 255).astype("uint8") for cyt in cyts]
    ncls = [(ncl * 255).astype("uint8") for ncl in ncls]
    
    # Get threshold
    cyts_thresh = threshold_otsu(np.concatenate(cyts, axis=0))
    ncls_thresh = threshold_otsu(np.concatenate(ncls, axis=0))
    
    # Process
    outputs = Parallel(n_jobs=-1)(
        delayed(_process)(i, cyt, ncl) 
        for i, (cyt, ncl) in enumerate(zip(cyts, ncls))
        )    
    cyts_med = [data[0] for data in outputs]
    ncls_med = [data[1] for data in outputs]
    cyts_msk = [data[2] for data in outputs]
    ncls_msk = [data[3] for data in outputs]

#%% Class : Display() ---------------------------------------------------------

class Display:
    
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.init_data()
        self.init_viewer()

    def init_data(self):
        
        # Load
        path = self.paths[self.idx]
        dir_path = data_path / path.stem
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        cyt_med_path = str(stk_path).replace("stk", "cyt_med")
        ncl_med_path = str(stk_path).replace("stk", "ncl_med")
        cyt_msk_path = str(stk_path).replace("stk", "cyt_msk")
        ncl_msk_path = str(stk_path).replace("stk", "ncl_msk")
        self.stk = io.imread(stk_path)
        self.C1 = self.stk[..., 0]
        self.C2 = self.stk[..., 1]
        self.C3 = self.stk[..., 2]
        self.C4 = self.stk[..., 3]
        self.cyt_msk = io.imread(cyt_msk_path) * 255
        self.ncl_msk = io.imread(ncl_msk_path) * 255
        self.cyt_med = io.imread(cyt_med_path)
        self.ncl_med = io.imread(ncl_med_path)
        
    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            self.C1, name="NRP2", visible=True,
            blending="additive", colormap="bop orange",
            gamma=0.75,
            )
        self.viewer.add_image(
            self.C2, name="virus", visible=False,
            blending="additive", colormap="green",
            )
        self.viewer.add_image(
            self.C3, name="EEA1", visible=False,
            blending="additive", colormap="magenta",
            )
        self.viewer.add_image(
            self.C4, name="nuclei", visible=True,
            blending="additive", colormap="bop blue",
            )
        self.viewer.add_image(
            self.cyt_msk, name="cyt_msk", visible=True,
            rendering="attenuated_mip", attenuation=0.5, colormap="bop orange",
            )
        self.viewer.add_image(
            self.ncl_msk, name="ncl_msk", visible=True,
            rendering="attenuated_mip", attenuation=0.5, colormap="bop blue",
            )
        
        # 3D display
        self.viewer.dims.ndisplay = 3
        
        # Create "select stack" menu
        self.stk_group_box = QGroupBox("Select stack")
        stk_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        stk_group_layout.addWidget(self.btn_next_image)
        stk_group_layout.addWidget(self.btn_prev_image)
        self.stk_group_box.setLayout(stk_group_layout)
        self.btn_next_image.clicked.connect(self.next_stack)
        self.btn_prev_image.clicked.connect(self.prev_stack)
        
        # Create texts
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_image.setText(
            f"{self.paths[self.idx].name}"
            )
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stk_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_stack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_stack()
        
    def update_stack(self):
        self.viewer.layers["NRP2"].data = self.C1
        self.viewer.layers["virus"].data = self.C2
        self.viewer.layers["EEA1"].data = self.C3
        self.viewer.layers["nuclei"].data = self.C4
        self.viewer.layers["cyt_msk"].data = self.cyt_msk
        self.viewer.layers["ncl_msk"].data = self.ncl_msk
        
    def update_text(self):
        self.info_image.setText(f"{self.paths[self.idx].name}")
        
    def next_stack(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.init_data()
            self.update_stack()
            self.update_text()
            
    def prev_stack(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_stack()
            self.update_text()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Preprocess --------------------------------------------------------------

    # Paths
    paths = []
    for path in stk_paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_stk.tif"))
        if not stk_path or not stk_path[0].exists() or overwrite["preprocess"]:
            paths.append(path)
    
    print("preprocess : ", end="", flush=True)
    t0 = time.time()
    
    # Execute
    Parallel(n_jobs=-1)(
        delayed(preprocess)(path, rf=rf) 
        for path in paths
        )  
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Process -----------------------------------------------------------------

    # Paths
    paths = []
    for path in stk_paths:
        dir_path = Path(data_path / path.stem)
        cyt_med_path = list(dir_path.glob("*_cyt_med.tif"))
        if not cyt_med_path or not cyt_med_path[0].exists() or overwrite["process"]:
            paths.append(path)

    print("process    : ", end="", flush=True)
    t0 = time.time()
    
    if paths:
        process(
            paths, 
            cyt_coeff=cyt_coeff,
            ncl_coeff=ncl_coeff,
            )
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display -----------------------------------------------------------------
    
    Display(stk_paths)

#%%

    # # Threshold
    # cyts_thresh = threshold_otsu(np.concatenate(cyts, axis=0))
    # ncls_thresh = threshold_otsu(np.concatenate(ncls, axis=0))
    # cyts_med_thresh = threshold_otsu(np.concatenate(cyts_med, axis=0))
    # ncls_med_thresh = threshold_otsu(np.concatenate(ncls_med, axis=0))
    
#%%

    # from skimage.filters import threshold_otsu
    # from skimage.morphology import remove_small_objects    
    
    # # Load data
    # stk = process(paths[4])
    # # voxsize = 
    
    # # Format data
    # cyt = (norm_pct(stk[..., 0]) * 255).astype("uint8")
    # ncl = (norm_pct(stk[..., 3]) * 255).astype("uint8")
    # cyt = median(cyt, footprint=ball(5))
    # ncl = median(ncl, footprint=ball(5))
    # cyt_thresh = threshold_otsu(cyt)
    # ncl_thresh = threshold_otsu(ncl)
    # cyt_msk = cyt > cyt_thresh
    # ncl_msk = ncl > ncl_thresh
    # cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
    # ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.dims.ndisplay = 3
    # viewer.add_image(
    #     cyt, name="cyt", visible=1,
    #     blending="additive", colormap="bop orange",
    #     )
    # viewer.add_image(
    #     ncl, name="nuclei", visible=1,
    #     blending="additive", colormap="bop blue",
    #     )
    # viewer.add_image(
    #     cyt_msk, name="cyt_msk", visible=0,
    #     rendering="attenuated_mip", attenuation=0.5, colormap="bop orange",
    #     )
    # viewer.add_image(
    #     ncl_msk, name="ncl_msk", visible=0,
    #     rendering="attenuated_mip", attenuation=0.5, colormap="bop blue",
    #     )