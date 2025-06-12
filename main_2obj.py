#%% Imports -------------------------------------------------------------------

import time
import pickle
import shutil
import napari
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# functions
# from functions import import_htk, prepare_htk,  
from functions import check_nd2, import_nd2, prepare_data, save_tif, load_data 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# skimage
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_local
from skimage.segmentation import watershed, relabel_sequential
from skimage.morphology import (
    ball, h_maxima, binary_dilation, remove_small_objects, remove_small_holes)

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

#%% Comments ------------------------------------------------------------------

'''
'''

#%% Inputs --------------------------------------------------------------------

warnings.filterwarnings("ignore")

procedure = {
    
    "extract" : 0,
    "predict" : 0,
    "process" : 0,
    "analyse" : 0,
    "display" : 1,
    
    }

parameters = {
    
    # Paths
       
    "rmt_path" : 
        Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data\\2OBJ"),
    "loc_path" : 
        Path("D:\local_Suzuki\data\\2OBJ"),
    "model_name_cyt" : 
        "model_512_normal_3000-1179_1_cyt",
    "model_name_ncl" : 
        "model_512_normal_3000-2034_1_ncl",
        
    "save" : "loc",
    
    # Parameters
        
        # extract
        "voxsize"     : 0.2,
        
        # process
        "cyt_thresh"  : 0.05, # 0.05
        "ncl_thresh"  : 0.15, # 0.15
        "blb_tcoeffs" : [2, 2, 2], # C1, C2, C3
    
    # Display
    "cmaps" : ["bop orange", "magenta", "green", "blue"],
    
    } 

#%% Class(Main) ---------------------------------------------------------------

class Main:
    
    def __init__(
            self, 
            procedure=procedure, 
            parameters=parameters,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        
        # Initialize
        self.save = self.parameters["save"]
        self.nd2_paths = list(self.parameters["rmt_path"].rglob("*.nd2"))
                
        # Run
        if self.procedure["extract"]:
            self.extract() 
        if self.procedure["predict"]:
            self.predict() 
        if self.procedure["process"]:
            self.process() 
        if self.procedure["analyse"]:
            self.analyse() 
                            
#%% Class(Main) : extract() ---------------------------------------------------
        
    def extract(self):
            
        def _extract(path):
            
            # Setup directory
            out_path = self.parameters[f"{self.save}_path"] / path.stem 
            C1_path = out_path / "C1.tif"
            if out_path.exists():
                if self.procedure["extract"] == 1 and C1_path.exists():
                    return
                elif self.procedure["extract"] == 2:
                    for item in out_path.iterdir():
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                    
            # Import
            shape = check_nd2(path)
            if shape[1] != 4:
                return
            
            t0 = time.time()
            print(f"{path.name} : ", end="", flush=False)
            
            metadata, htk = import_nd2(
                path, voxsize=self.parameters["voxsize"]) 
            
            # Save
            
            with open(str(out_path / "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
                
            metadata_df = pd.DataFrame(
                list(metadata.items()), columns=['Key', 'Value'])
            metadata_df.to_csv(out_path / "metadata.csv", index=False)
            
            for c in range(htk.shape[1]):
                save_tif(
                    htk[:, c, ...], out_path / f"C{c + 1}.tif", 
                    voxsize=metadata["vsize1"][0],
                    )
                
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
                
        # Serial --------------------------------------------------------------

        print("extract() : ")
        
        for path in self.nd2_paths:
            _extract(path)
        
#%% Class(Main) : predict() ---------------------------------------------------
    
    def predict(self):
        
        print("predict() : ")
        
        # Initialize
        dir_list = self.parameters[f"{self.save}_path"].iterdir()
        out_paths = [f for f in dir_list if f.is_dir()]
        unet_cyt = UNet(load_name=self.parameters["model_name_cyt"])
        unet_ncl = UNet(load_name=self.parameters["model_name_ncl"])
        
        for out_path in out_paths:

            # Load data
            cyt_prd_path = out_path / "cyt_prd.tif"
            if cyt_prd_path.exists() and self.procedure["predict"] == 1:
            
                continue
            
            else:
                
                t0 = time.time()
                print(f"{out_path.name} : ", end="", flush=False)
                
                data = load_data(out_path)
                
                # Prepare
                prp_cyt = prepare_data(
                    data["htk"][:, 0, ...],
                    data["htk"][:, 3, ...],
                    )
                prp_cyt = norm_pct(prp_cyt)
                prp_ncl = norm_pct(data["htk"][:, 3, ...])
                
                # Predict
                cyt_prd = unet_cyt.predict(prp_cyt, verbose=0)
                ncl_prd = unet_ncl.predict(prp_ncl, verbose=0)
                
                # Save
                cyt_prd = (cyt_prd * 255).astype("uint8")
                ncl_prd = (ncl_prd * 255).astype("uint8")
                save_tif(
                    cyt_prd, out_path / "cyt_prd.tif", 
                    voxsize=data["metadata"]["vsize1"][0],
                    )     
                save_tif(
                    ncl_prd, out_path / "ncl_prd.tif", 
                    voxsize=data["metadata"]["vsize1"][0],
                    )      
            
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):
        
        def get_masks(htk, cyt_prd, ncl_prd):
            
            # Process
            cyt_prd_prp = norm_pct(gaussian(cyt_prd, sigma=1))
            ncl_prd_prp = norm_pct(gaussian(ncl_prd, sigma=1))
            cyt_prp = norm_pct(gaussian(htk[:, 0, ...], sigma=1))
            ncl_prp = norm_pct(gaussian(htk[:, 3, ...], sigma=1))
            cyt_prp *= cyt_prd_prp
            ncl_prp *= ncl_prd_prp
            cyt_msk = cyt_prp > self.parameters["cyt_thresh"]
            ncl_msk = ncl_prp > self.parameters["ncl_thresh"]
            cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
            ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
            cyt_msk = remove_small_holes(cyt_msk, area_threshold=1e5)
            ncl_msk = remove_small_holes(ncl_msk, area_threshold=1e5)
            
            # Posprocess ncl_msk
            ncl_lbl = label(ncl_msk)
            for props in regionprops(ncl_lbl, intensity_image=cyt_msk):
                lbl = props.label
                val = props.intensity_mean
                if val < 0.25:
                    ncl_lbl[ncl_lbl == lbl] = 0
            ncl_msk = ncl_lbl > 0
            
            # Postprocess cyt_msk
            cyt_msk[ncl_msk] = 1
            
            return cyt_msk, ncl_msk
        
        def get_blobs(
                arr, mask=None, 
                sigma0=0.5, sigma1=5, thresh_coeff=2, 
                out_prp=False
                ):
            
            # preprocess
            gbl0 = gaussian(arr, sigma=sigma0)
            gbl1 = gaussian(arr, sigma=sigma1)   
            prp = (gbl0 - gbl1) / gbl1
            prp = norm_pct(prp, pct_low=0, pct_high=100)
            local_thresh = threshold_local(prp, 21, offset=0)
            msk = prp > local_thresh * 2
            if mask is not None:
                msk[mask == 0] = 0        
            
            # watershed
            hmax = h_maxima(prp, 0.1, footprint=ball(1))
            hmax = (hmax * 255).astype("uint8")
            hmax = label(hmax)
            lbl = watershed(-prp, markers=hmax, mask=msk)
            
            # relabel
            lbl = relabel_sequential(lbl)[0]
            
            if out_prp:
                return lbl, prp
            else:
                return lbl
                        
        def _process(out_path):

            # Load data
            cyt_msk_path = out_path / "ncl_msk.tif"
        
            if cyt_msk_path.exists() and self.procedure["process"] == 1:
            
                return
            
            else:
                
                data = load_data(out_path)
                
                # Get masks
                cyt_msk, ncl_msk = get_masks(
                    data["htk"], data["cyt_prd"], data["ncl_prd"])
                
                for c in range(3):
                    
                    # Get blobs
                    blb_lbl = get_blobs(
                        data["htk"][:, c, ...], mask=cyt_msk, 
                        sigma0=1, sigma1=5, 
                        thresh_coeff=self.parameters["blb_tcoeffs"][c]
                        )
                            
                    # Save
                    save_tif(
                        blb_lbl.astype("uint16"), 
                        out_path / f"C{c + 1}_lbl.tif", 
                        voxsize=data["metadata"]["vsize1"][0],
                        )   
                
                # Save
                save_tif(
                    (cyt_msk * 255).astype("uint8"), 
                    out_path / "cyt_msk.tif", 
                    voxsize=data["metadata"]["vsize1"][0],
                    )   
                save_tif(
                    (ncl_msk * 255).astype("uint8"), 
                    out_path / "ncl_msk.tif", 
                    voxsize=data["metadata"]["vsize1"][0],
                    )   

        # ---------------------------------------------------------------------
        
        # Initialize
        dir_list = self.parameters[f"{self.save}_path"].iterdir()
        out_paths = [f for f in dir_list if f.is_dir()]
        
        t0 = time.time()
        print("process() : ", end="", flush=False)

        Parallel(n_jobs=-1)(
            delayed(_process)(out_path) 
            for out_path in out_paths
            ) 

        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self): 
        
        def get_blobs_int(lbl, img, return_idx=False):
            lbl_r = lbl.ravel()
            img_r = img.ravel()        
            counts = np.bincount(lbl_r)
            sum_img = np.bincount(lbl_r, weights=img_r)
            idx = np.nonzero(counts)[0]
            idx = idx[idx != 0]  
            if return_idx:
                return (idx, sum_img[idx] / counts[idx])
            else:
                return sum_img[idx] / counts[idx]
        
        def _analyse(out_path):
            
            # Load data
            C2_lbl_v_path = out_path / "C2_lbl_v.tif"
        
            if C2_lbl_v_path.exists() and self.procedure["analyse"] == 1:
            
                return
            
            else:
                
                data = load_data(out_path)
                
                # Initialize
                C2_lbl = data["C2_lbl"]
                metadata = data["metadata"]
                lbls = np.arange(1, np.max(C2_lbl) + 1)
                cyt_edt = get_edt(data["cyt_msk"], rescale_factor=0.5)
                
                metadata = data["metadata"]
                
                # Append results
                C2_results = {
                    
                    "lbl"         : lbls,
                    "path"        : metadata["path"],
                    "date"        : metadata["date"],
                    "rep"         : metadata["rep"],
                    "cell"        : metadata["cell"],
                    "time"        : metadata["time"],
                    "C1_name"     : metadata["chn_names"][0],
                    "C2_name"     : metadata["chn_names"][1],
                    "C3_name"     : metadata["chn_names"][2],
                    "C4_name"     : metadata["chn_names"][3],
                    "cond"        : metadata["cond"],
                    "exp"         : metadata["exp"],
                    "count"       : [(C2_lbl == l).sum() for l in lbls],
                    "cyt_edt_avg" : get_blobs_int(C2_lbl, cyt_edt),
                    "ncl_msk_avg" : get_blobs_int(C2_lbl, data["ncl_msk"] > 0),
                    "C1_avg"      : get_blobs_int(C2_lbl, data["htk"][:, 0, ...]),
                    "C2_avg"      : get_blobs_int(C2_lbl, data["htk"][:, 1, ...]),
                    "C3_avg"      : get_blobs_int(C2_lbl, data["htk"][:, 2, ...]),
                    "C1_msk_avg"  : get_blobs_int(C2_lbl, data["C1_lbl"] > 0),
                    "C3_msk_avg"  : get_blobs_int(C2_lbl, data["C3_lbl"] > 0),
                    
                    }
                
                # Get valid labels 
                lbls_v = lbls[C2_results["C3_msk_avg"] == 0]
                C2_lbl_v = C2_lbl.copy()
                C2_lbl_v[~np.isin(C2_lbl_v, lbls_v)] = 0
                
                # Save
                
                C2_results = pd.DataFrame(C2_results)   
                C2_results_v = C2_results[C2_results["C3_msk_avg"] == 0]
                C2_results.to_csv(out_path / "C2_results.csv", index=False)
                C2_results_v.to_csv(out_path / "C2_results_v.csv", index=False)
                
                save_tif(
                    C2_lbl_v.astype("uint16"), 
                    out_path / "C2_lbl_v.tif", 
                    voxsize=data["metadata"]["vsize1"][0],
                    )  
                        
        # ---------------------------------------------------------------------
        
        # Initialize
        dir_list = self.parameters[f"{self.save}_path"].iterdir()
        out_paths = [f for f in dir_list if f.is_dir()]
        
        t0 = time.time()
        print("analyse() : ", end="", flush=False)
                
        Parallel(n_jobs=-1)(
            delayed(_analyse)(out_path) 
            for out_path in out_paths
            ) 
                
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Merge & save C2_results
        
        dir_path = self.parameters[f"{self.save}_path"]
        
        C2_results_m = []
        for path in list(dir_path.rglob("*C2_results.csv")):
            C2_results_m.append(pd.read_csv(path))
        C2_results_m = pd.concat(C2_results_m, ignore_index=True)
        C2_results_m.to_csv(
            dir_path / "C2_results_m.csv", 
            index=False,
            )
        
        C2_results_v_m = []
        for path in list(dir_path.rglob("*C2_results_v.csv")):
            C2_results_v_m.append(pd.read_csv(path))
        C2_results_v_m = pd.concat(C2_results_v_m, ignore_index=True)
        C2_results_v_m.to_csv(
            dir_path / "C2_results_v_m.csv", 
            index=False,
            )

#%% Class(Display) ------------------------------------------------------------
    
class Display:
    
    def __init__(
            self, 
            procedure=procedure, 
            parameters=parameters,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        
        # Initialize
        self.idx = 0
        self.save = self.parameters["save"]
        dir_list = self.parameters[f"{self.save}_path"].iterdir()
        self.out_paths = [f for f in dir_list if f.is_dir()]
                
        # Run
        if self.procedure["display"]:
            self.load_data()
            self.init_viewer()
            self.show_masks()
        
#%% Class(Display) : function(s) ----------------------------------------------
                
    def load_data(self):
        
        def get_outlines(arr):
            arr = arr > 0
            out = []
            for img in arr:
                out.append(binary_dilation(img) ^ img)
            return np.stack(out)
         
        self.data = load_data(self.out_paths[self.idx])
        self.data["cyt_out"] = get_outlines(self.data["cyt_msk"])
        self.data["ncl_out"] = get_outlines(self.data["ncl_msk"])
        for c in range(3):
            self.data[f"C{c + 1}_out"] = get_outlines(
                self.data[f"C{c + 1}_lbl"] > 0)
        
    def next_hstack(self):
        if self.idx < len(self.out_paths) - 1:
            self.idx += 1
            self.load_data()
            self.update()
            
    def prev_hstack(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_data()
            self.update()
                    
    def center_view(self):
        for name in self.viewer.layers:
            name = str(name)
            shape = self.viewer.layers[name].data.shape
            self.viewer.camera.center = (
                shape[0] // 2, shape[1] // 2, shape[2] // 2)
            self.viewer.camera.zoom = 1.0
            
    def hide_outlines(self):
        for name in self.viewer.layers:
            name = str(name)
            if self.rad_chk_C1b.isChecked(): tag = "C1"
            if self.rad_chk_C2b.isChecked(): tag = "C2"
            if self.rad_chk_C3b.isChecked(): tag = "C3"
            if f"{tag}_out" in name:
                self.viewer.layers[name].visible = 0
                
    def show_outlines(self):
        for name in self.viewer.layers:
            name = str(name)
            if self.rad_chk_C1b.isChecked(): tag = "C1"
            if self.rad_chk_C2b.isChecked(): tag = "C2"
            if self.rad_chk_C3b.isChecked(): tag = "C3"
            if f"{tag}_out" in name:
                self.viewer.layers[name].visible = 1
            
    def show_masks(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if name in ["cyt_msk", "ncl_msk", "C1_msk", "C2_msk_v"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.center_view()
    
    def show_cyt_prd(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C1", "cyt_prd"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.center_view()
        
    def show_ncl_prd(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C4", "ncl_prd"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.center_view()
                
    def show_chk(self, tag="C1"):
        self.viewer.dims.ndisplay = 2
        for name in self.viewer.layers:
            name = str(name)
            if name in [f"{tag}", f"{tag}_out", "cyt_out", "ncl_out"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.center_view()

#%% Class(Display) : init_viewer() --------------------------------------------                

    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Create "hstack" menu
        self.htk_group_box = QGroupBox("Select hstack")
        htk_group_layout = QVBoxLayout()
        self.btn_next_htk = QPushButton("next")
        self.btn_prev_htk = QPushButton("prev")
        htk_group_layout.addWidget(self.btn_next_htk)
        htk_group_layout.addWidget(self.btn_prev_htk)
        self.htk_group_box.setLayout(htk_group_layout)
        self.btn_next_htk.clicked.connect(self.next_hstack)
        self.btn_prev_htk.clicked.connect(self.prev_hstack)
                
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QHBoxLayout()
        self.rad_masks = QRadioButton("mask")
        self.rad_cyt_prd = QRadioButton("predict cyt")
        self.rad_ncl_prd = QRadioButton("predict ncl")
        self.rad_masks.setChecked(True)
        dsp_group_layout.addWidget(self.rad_masks)
        dsp_group_layout.addWidget(self.rad_cyt_prd)
        dsp_group_layout.addWidget(self.rad_ncl_prd)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_masks.toggled.connect(
            lambda checked: self.show_masks() if checked else None)
        self.rad_cyt_prd.toggled.connect(
            lambda checked: self.show_cyt_prd() if checked else None)
        self.rad_ncl_prd.toggled.connect(
            lambda checked: self.show_ncl_prd() if checked else None)
        
        # Create "check" menu
        self.chk_group_box = QGroupBox("Check blobs")
        chk_group_layout = QHBoxLayout()
        self.rad_chk_C1b = QRadioButton("C1")
        self.rad_chk_C2b = QRadioButton("C2")
        self.rad_chk_C3b = QRadioButton("C3")
        chk_group_layout.addWidget(self.rad_chk_C1b)
        chk_group_layout.addWidget(self.rad_chk_C2b)
        chk_group_layout.addWidget(self.rad_chk_C3b)
        self.chk_group_box.setLayout(chk_group_layout)
        self.rad_chk_C1b.toggled.connect(
            lambda checked: self.show_chk(tag="C1") if checked else None)
        self.rad_chk_C2b.toggled.connect(
            lambda checked: self.show_chk(tag="C2") if checked else None)
        self.rad_chk_C3b.toggled.connect(
            lambda checked: self.show_chk(tag="C3") if checked else None)
                
        # Create texts
        self.info_path = QLabel()
        self.info_path.setFont(QFont("Consolas"))
        self.info_path.setText(self.get_info())

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.htk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addWidget(self.chk_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_path)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter")
        self.init_layers()        
        
        # Shortcuts
        
        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_hstack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_hstack()
            
        @self.viewer.bind_key("Backspace", overwrite=True)
        def toggle_outlines(viewer):
            self.hide_outlines()
            yield
            self.show_outlines()
                                                
#%% Class(Display) : init_layers() --------------------------------------------            

    def init_layers(self):
        
        self.cmaps = self.parameters["cmaps"]
        
        for c in range(self.data["htk"].shape[1]- 1, -1, -1):
            
            # htk
            self.viewer.add_image(
                self.data["htk"][:, c, ...], visible=0,
                name=f"C{c + 1}", colormap=self.cmaps[c], 
                blending="additive", gamma=0.5, 
                )
            
            # blobs
            
            if c == 1:
                
                self.viewer.add_image(
                    self.data["C2_lbl_v"] > 0, visible=0,
                    name="C2_msk_v", colormap=self.cmaps[c],
                    blending="additive", opacity=0.75, 
                    rendering="attenuated_mip", attenuation=0.5,  
                    )
                
            if c < 3:
                
                self.viewer.add_image(
                    self.data[f"C{c + 1}_lbl"] > 0, visible=0,
                    name=f"C{c + 1}_msk", colormap=self.cmaps[c],
                    blending="additive", opacity=0.75, 
                    rendering="attenuated_mip", attenuation=0.5,  
                    )
                
                self.viewer.add_image(
                    self.data[f"C{c + 1}_out"], visible=0,
                    name=f"C{c + 1}_out", colormap="gray",
                    blending="additive", opacity=0.5,  
                    ) 

        # masks
        
        self.viewer.add_image(
            self.data["ncl_msk"], visible=0,
            name="ncl_msk", colormap="blue",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        self.viewer.add_image(
            self.data["ncl_out"], visible=0,
            name="ncl_out", colormap="gray",
            blending="additive", opacity=0.2,  
            )
        
        self.viewer.add_image(
            self.data["cyt_msk"], visible=0,
            name="cyt_msk", colormap="gray",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        self.viewer.add_image(
            self.data["cyt_out"], visible=0,
            name="cyt_out", colormap="gray",
            blending="additive", opacity=0.2,  
            )
        
        # predictions
        
        self.viewer.add_image(
            self.data["cyt_prd"], visible=0,
            name="cyt_prd", colormap="turbo",
            blending="translucent_no_depth", opacity=0.5,
            rendering="attenuated_mip", attenuation=0.5, 
            )

        self.viewer.add_image(
            self.data["ncl_prd"], visible=0,
            name="ncl_prd", colormap="turbo",
            blending="translucent_no_depth", opacity=0.5,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        # Adjust viewer
        self.viewer.dims.ndisplay = 3
        self.center_view()
        
#%% Class(Display) : get_info() -----------------------------------------------

    def get_info(self):
        
        def format_tuple(tpl, fmt=".3f", sep=", "):
            fmt_tpl = []
            for elm in tpl:
                fmt_tpl.append(f"{elm:{fmt}}")
            return sep.join(fmt_tpl)
        
        # ---------------------------------------------------------------------
        
        metadata = self.data['metadata']
        
        # Formatting
        chn_names = [
            f"C{c + 1}   : {metadata['chn_names'][c]}\n"
            for c in range(self.data["htk"].shape[1])
            ]
        shape0 = format_tuple(metadata["shape0"], fmt="04d")
        shape1 = format_tuple(metadata["shape1"], fmt="04d")
        vsize0 = format_tuple(metadata["vsize0"], fmt=".3f")
        vsize1 = format_tuple(metadata["vsize1"], fmt=".3f")
        
        return (
            
            f"{self.data['metadata']['path'].name}"
            
            "\n\n"
            f"cond : {metadata['cond']}" 
            "\n"
            f"{''.join(chn_names)}"
            
            "\n"
            f"shape0 = ({shape0})"
            "\n"
            f"shape1 = ({shape1})"
            "\n"
            f"vsize0 = ({vsize0})"
            "\n"
            f"vsize1 = ({vsize1})"
            "\n"
   
            )        
        
#%% Class(Display) : update() -------------------------------------------------  

    def update(self):
        
        
        for c in range(self.data["htk"].shape[1]):
            
            # blobs
            
            if c == 1:
            
                self.viewer.layers["C2_msk_v"].data = (
                    self.data["C2_lbl_v"] > 0)                
            
            if c < 3: 

                self.viewer.layers[f"C{c + 1}_msk"].data = (
                    self.data[f"C{c + 1}_lbl"] > 0)
                self.viewer.layers[f"C{c + 1}_out"].data = (
                    self.data[f"C{c + 1}_out"] > 0)
            
            # htk
            self.viewer.layers[f"C{c + 1}"].data = (
                self.data["htk"][:, c, ...])

        # masks
        self.viewer.layers["ncl_msk"].data = self.data["ncl_msk"]
        self.viewer.layers["cyt_msk"].data = self.data["cyt_msk"]
        self.viewer.layers["ncl_out"].data = self.data["ncl_out"]
        self.viewer.layers["cyt_out"].data = self.data["cyt_out"]
        
        # predictions
        self.viewer.layers["cyt_prd"].data = self.data["cyt_prd"]
        self.viewer.layers["ncl_prd"].data = self.data["ncl_prd"]
        
        # Adjust viewer
        self.center_view()
        
        # Update info
        self.info_path.setText(self.get_info())

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":    
    main = Main()
    display = Display()
