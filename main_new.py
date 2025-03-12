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

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.filters.rank import median
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
    "preprocess" : 0,
    }

# Parameters
rf = 0.5

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))

#%% Function : preprocess() ---------------------------------------------------

def preprocess(path, rf=0.5):
        
    with nd2.ND2File(path) as ndfile:
        
        # voxSize
        voxsize0 = (
            ndfile.voxel_size()[2],
            ndfile.voxel_size()[1],
            ndfile.voxel_size()[0],
            )
        
        # Determine isotropic rescaling factor (rfi)
        rfi = voxsize0[1] / voxsize0[0]
        
        # Load & rescale stack
        stk = ndfile.asarray()
        stk = rescale(stk, (1, 1, rfi, rfi), order=0) # iso rescale (rfi)
        stk = rescale(stk, (rf, 1, rf, rf), order=0) # custom rescale (rf)
            
        # Flip z axis
        stk = np.flip(stk, axis=0)
        
        # Adjust voxSize
        voxsize1 = (
            voxsize0[0] / rf, 
            voxsize0[1] / (rfi * rf),
            voxsize0[2] / (rfi * rf),
            )
        
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
        stk_name = path.stem + f"_{voxsize1[0]}_stk.tif"
        resolution = (1 / voxsize1[1], 1 / voxsize1[2])
        metadata = {"axes" : "ZCYX", "spacing" : voxsize1[0], "unit" : "um"}
        tifffile.imwrite(
            dir_path / stk_name, stk, 
            imagej=True, resolution=resolution, metadata=metadata
            )

#%% Function : process() ------------------------------------------------------

def process(paths):

    # global cyts, ncls   
    
    def _process(stk):
        med = median(stk, footprint=ball(5))
        return med
        
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
    
    
    cyts_med = Parallel(n_jobs=-1)(
        delayed(_process)(cyt) 
        for cyt in cyts
        )
    ncls_med = Parallel(n_jobs=-1)(
        delayed(_process)(ncl) 
        for ncl in ncls
        )  
    
    return cyts_med, ncls_med   
    
    

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
        self.stk = io.imread(stk_path)
        self.C1 = self.stk[..., 0]
        self.C2 = self.stk[..., 1]
        self.C3 = self.stk[..., 2]
        self.C4 = self.stk[..., 3]
        
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
        if not stk_path or not stk_path[0] or overwrite["preprocess"]:
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

    print("process    : ", end="", flush=True)
    t0 = time.time()
    
    cyts_med, ncls_med = process(stk_paths)
    
    # # Execute
    # Parallel(n_jobs=-1)(
    #     delayed(process)(path) 
    #     for path in paths
    #     )  
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display -----------------------------------------------------------------
    
    # Display(stk_paths)
    
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