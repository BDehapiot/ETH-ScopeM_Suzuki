#%% Imports -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

# matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Comments ------------------------------------------------------------------

'''
- count
- cyt_edt_avg
- ncl_msk_avg
- C1_avg
- C2_avg
- C3_avg
- C1_msk_avg
- C3_msk_avg
'''

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Paths
    # "data_path" : Path("D:\local_Suzuki\data"),
    "data_path" : Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data"),
    "tags"      : ["2OBJ"],
    
    # Parameters
    
    "data_x"    : "cyt_edt_avg",
    "data_y"    : "C1_avg",
    "tag0_in"   : ["Im00", "IM00", "N00"],
    "tag0_out"  : [],
    "tag1_in"   : ["Dr01"],
    "tag1_out"  : [],
    
    }

#%% Function(s) ---------------------------------------------------------------

def filter_data(df, name, tag_in, tag_out):
    if tag_in:
        mask_in = df["path"].apply(
            lambda x: any(tag in x for tag in tag_in))
    else:
        mask_in = pd.Series(True, index=df.index)
    if tag_out:
        mask_out = df["path"].apply(
            lambda x: any(tag in x for tag in tag_out))
    else:
        mask_out = pd.Series(False, index=df.index)
    return df.loc[mask_in & ~mask_out, name]

#%% Class(Plot) ---------------------------------------------------------------

class Plot:
    
    def __init__(self, parameters=parameters):
        
        # Fetch
        self.data_path = parameters["data_path"]
        self.tags      = parameters["tags"]
        self.data_x    = parameters["data_x"]
        self.data_y    = parameters["data_y"]
        self.tag0_in   = parameters["tag0_in"]
        self.tag0_out  = parameters["tag0_out"]
        self.tag1_in   = parameters["tag1_in"]
        self.tag1_out  = parameters["tag1_out"]
        
        # Run
        self.initialize()
        self.load()
        # self.all_vs_valid()
        self.cond0_vs_cond1()

#%% Class(Plot) : initialize() ------------------------------------------------

    def initialize(self):

        if "2OBJ" in self.tags:
            self.exp = "2OBJ"
        if "3OBJ" in self.tags:
            self.exp = "3OBJ"
        
#%% Class(Plot) : load() ------------------------------------------------------

    def load(self):
        
        self.df_m = pd.read_csv(
            self.data_path / self.exp / "C2_results_m.csv")
        self.df_v_m = pd.read_csv(
            self.data_path / self.exp / "C2_results_v_m.csv")
        
#%% Class(Plot) : all_vs_valid() ----------------------------------------------

    def all_vs_valid(self):
        
        # Filter data
        x_m = filter_data(
            self.df_m, self.data_x, self.tag0_in, self.tag0_out)
        y_m = filter_data(
            self.df_m, self.data_y, self.tag0_in, self.tag0_out)
        x_v_m = filter_data(
            self.df_v_m, self.data_x, self.tag0_in, self.tag0_out)
        y_v_m = filter_data(
            self.df_v_m, self.data_y, self.tag0_in, self.tag0_out)
    
        # Initialize plot
        fig = plt.figure(figsize=(6, 9), layout="tight")
        gs = GridSpec(3, 2, figure=fig)
    
        # Scatter plot
        
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.scatter(x_m, y_m, label="all", s=5)
        ax0.scatter(x_v_m, y_v_m, label="valid", s=5)
        
        ax0.set_title(f"{self.data_x} vs. {self.data_y}")
        ax0.set_ylabel(self.data_y)
        ax0.set_xlabel(self.data_x)
        ax0.legend(loc="upper right")
        
        # Box plots
        
        boxplot_params = {"widths" : 0.6, "showfliers" : False}
        
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.boxplot(y_m, positions=[0], tick_labels=["all"], **boxplot_params)
        ax1.boxplot(y_v_m, positions=[1], tick_labels=["valid"], **boxplot_params)
        ax1.set_title(self.data_y)
        ax1.set_ylabel(self.data_y)   
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.boxplot(x_m, positions=[0], tick_labels=["all"], **boxplot_params)
        ax2.boxplot(x_v_m, positions=[1], tick_labels=["valid"], **boxplot_params)
        ax2.set_title(self.data_x)
        ax2.set_ylabel(self.data_x)   

#%% Class(Plot) : cond0_vs_cond1() --------------------------------------------

    def cond0_vs_cond1(self):
        
        # Filter data
        x0_v_m = filter_data(
            self.df_v_m, self.data_x, self.tag0_in, self.tag0_out)
        y0_v_m = filter_data(
            self.df_v_m, self.data_y, self.tag0_in, self.tag0_out)
        x1_v_m = filter_data(
            self.df_v_m, self.data_x, self.tag1_in, self.tag1_out)
        y1_v_m = filter_data(
            self.df_v_m, self.data_y, self.tag1_in, self.tag1_out)
        
        # Initialize plot
        fig = plt.figure(figsize=(6, 9), layout="tight")
        gs = GridSpec(3, 2, figure=fig)
        
        # Info
        info = (   
            
            f"cond0 n : {len(x0_v_m)}\n"
            f"cond0 tag(s) in  : {','.join(self.tag0_in )}\n"
            f"cond0 tag(s) out : {','.join(self.tag0_out)}\n"
            f"------------------\n"
            f"cond1 n : {len(x1_v_m)}\n"
            f"cond1 tag(s) in  : {','.join(self.tag1_in )}\n"
            f"cond1 tag(s) out : {','.join(self.tag1_out)}\n"
            
            )
        
        # Scatter plot
        
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.scatter(x0_v_m, y0_v_m, label="cond0", s=5, alpha=0.5)
        ax0.scatter(x1_v_m, y1_v_m, label="cond1", s=5, alpha=0.5)
        
        ax0.set_title(f"{self.data_x} vs. {self.data_y}")
        ax0.set_ylabel(self.data_y)
        ax0.set_xlabel(self.data_x)
        ax0.legend(loc="upper right")
        
        ax0.text(
            0.025, 0.95, info, size=10, color="k",
            transform=ax0.transAxes, ha="left", va="top", 
            fontfamily="Consolas",
            )
        
        # Box plots
        
        boxplot_params = {"widths" : 0.6, "showfliers" : False}
        
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.boxplot(y0_v_m, positions=[0], tick_labels=["cond0"], **boxplot_params)
        ax1.boxplot(y1_v_m, positions=[1], tick_labels=["cond1"], **boxplot_params)
        ax1.set_title(self.data_y)
        ax1.set_ylabel(self.data_y)   
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.boxplot(x0_v_m, positions=[0], tick_labels=["cond0"], **boxplot_params)
        ax2.boxplot(x1_v_m, positions=[1], tick_labels=["cond1"], **boxplot_params)
        ax2.set_title(self.data_x)
        ax2.set_ylabel(self.data_x)  

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    plot = Plot()
    df_m = plot.df_m
    df_v_m = plot.df_v_m



#%%

    # def filter_data(df, name, tag_in, tag_out):
    #     if tag_in:
    #         mask_in = df["path"].apply(
    #             lambda x: any(tag in x for tag in tag_in))
    #     else:
    #         mask_in = pd.Series(True, index=df.index)
    #     if tag_out:
    #         mask_out = df["path"].apply(
    #             lambda x: any(tag in x for tag in tag_out))
    #     else:
    #         mask_out = pd.Series(False, index=df.index)
    #     return df.loc[mask_in & ~mask_out, name]
    
    # test = filter_data(df_v_m, )