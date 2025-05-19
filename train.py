#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.annotate import Annotate
from bdtools.models.unet import UNet

# functions
from functions import check_nd2, import_nd2, prepare_data

#%% Inputs --------------------------------------------------------------------

# Procedure
annotate = 1
train = 0
predict = 0

# Annotate
ncl = 1

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 0
load_name = "model_512_normal_3000-1683_1"

# preprocess
patch_size = 512
patch_overlap = 256
img_norm = "none"
msk_type = "normal"

# augment
iterations = 3000
gamma_p = 0.5
gblur_p = 0
noise_p = 0.5 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 8
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0001
patience = 20

# predict
idx = 80
voxsize = 0.2

#%% Initialize ----------------------------------------------------------------

data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data")
if ncl:
    train_path = Path("data", "train_ncl")
else:
    train_path = Path("data", "train")

#%% Function(s) ---------------------------------------------------------------

def format_data(imgs, msks):
    nYs = [img.shape[0] for img in imgs]
    nXs = [img.shape[1] for img in imgs]
    nY_max = np.max(nYs)
    nX_max = np.max(nXs)
    imgs_pad, msks_pad = [], []
    for img, msk in zip(imgs, msks):
        img = norm_pct(img) 
        nY, nX = img.shape
        y_pad = nY_max - nY if nY < nY_max else 0
        x_pad = nX_max - nX if nX < nX_max else 0
        pad_width = (
            (y_pad // 2, y_pad // 2), 
            (x_pad // 2, x_pad // 2),
            )
        imgs_pad.append(np.pad(img, pad_width, mode='reflect'))
        msks_pad.append(np.pad(msk, pad_width, mode='reflect'))
    imgs_pad = np.stack(imgs_pad)
    msks_pad = np.stack(msks_pad)
    return imgs_pad, msks_pad

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%% Annotate ------------------------------------------------------------------
    
    if annotate:
        
        Annotate(train_path)
    
#%% Train ---------------------------------------------------------------------
    
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(train_path.rglob("*.tif")):
            if "mask" in path.name:
                if Path(str(path).replace("_mask", "")).exists():
                    msks.append(io.imread(path))   
                    imgs.append(io.imread(str(path).replace("_mask", "")))

        # Format data
        imgs, msks = format_data(imgs, msks)

        unet = UNet(
            save_name="",
            load_name=load_name,
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        # Train
        unet.train(
            
            imgs, msks, 
            X_val=None, y_val=None,
            preview=preview,
            
            # Preprocess
            img_norm=img_norm, 
            msk_type=msk_type, 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscale_factor, 
            
            # Augment
            iterations=iterations,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )
        
#%% Predict -------------------------------------------------------------------
        
    if predict:
        
        # Get path
        nd2_path = list(data_path.rglob("*.nd2"))[idx]

        # Import nd2
        shape = check_nd2(nd2_path)
        if shape[1] == 4:
            
            print(nd2_path.name)
            
            # Import & prepare data
            t0 = time.time()
            print("load : ", end="", flush=False)
            _, C1 = import_nd2(nd2_path, z="all", c=0, voxsize=voxsize)
            _, C4 = import_nd2(nd2_path, z="all", c=3, voxsize=voxsize)
            prp = prepare_data(C1, C4)
            prp = norm_pct(prp)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
            # Predict
            t0 = time.time()
            print("predict : ", end="", flush=False)
            unet = UNet(load_name=load_name)
            prd = unet.predict(prp, verbose=0)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
            # Display
            viewer = napari.Viewer()
            viewer.add_image(prp)
            viewer.add_image(prd, 
                blending="additive", colormap="inferno", opacity=0.5
                )  
            
        else:
            print("Targeted nd2 file is not 4 channels")