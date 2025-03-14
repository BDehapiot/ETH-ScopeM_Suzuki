#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import format_stack, prepare_stack

# bdtools
from bdtools.models.annotate import Annotate
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Procedure
annotate = 0
train = 1
predict = 0

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 3

# UNet train()
preview = False

# preprocess
patch_size = 512
patch_overlap = 0
img_norm = "none"
msk_type = "normal"

# augment
iterations = 2000
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
learning_rate = 0.0005
patience = 20

# predict
rf = 0.5

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))   
train_path = Path("data", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    if annotate:
        Annotate(train_path)
    
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(train_path.glob("*.tif")):
            if "mask" in path.name:
                msks.append(io.imread(path))   
                imgs.append(io.imread(str(path).replace("_mask", "")))
        imgs = np.stack(imgs)
        msks = np.stack(msks)
         
        unet = UNet(
            save_name="",
            load_name="",
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
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
        
    if predict:
        
        # Format and merge stack
        path = stk_paths[1]
        stk, voxsize = format_stack(path, rf=rf)
        mrg = prepare_stack(stk)
        
        # Predict
        unet = UNet(
            load_name="model_512_normal_1000-160_2",
            )
        prd = unet.predict(mrg, verbose=3)
                
        # Display
        viewer = napari.Viewer()
        viewer.add_image(mrg)
        viewer.add_image(prd)