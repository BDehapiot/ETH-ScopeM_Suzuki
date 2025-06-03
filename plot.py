#%% Imports -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

# matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Comments ------------------------------------------------------------------

# data
'''
count
cyt_edt_avg
ncl_msk_avg
C1_avg
C2_avg
C3_avg
C1_msk_avg
C3_msk_avg
'''

# NRP2
'''
"E00" : "NRP2KO-eGFP",
"N00" : "NRP2KO-NRP2-eGFP",
"N01" : "NRP2KO-NRP2-eGFP_T319R",
"N02" : "NRP2KO-NRP2-eGFP_AAA",
"N03" : "NRP2KO-NRP2-eGFP_dA1A2",
"N04" : "NRP2KO-NRP2-eGFP_dB1",
"N05" : "NRP2KO-NRP2-eGFP_dB2",
"N06" : "NRP2KO-NRP2-eGFP_dMAM",
"N07" : "NRP2KO-NRP2-eGFP_dCyto",
"N08" : "NRP2KO-NRP2-eGFP_dSA",
"N09" : "NRP2KO-NRP2-eGFP_dA1A2B1B2",
"N10" : "NRP2KO-NRP2-eGFP_dB1B2",
"N11" : "NRP2KO-NRP2-eGFP_dSAB1",
"N12" : "NRP2KO-NRP2-eGFP_dSAB1B2",
'''

# Drugs
'''
"IM00" : "none",
"Dr01" : "DMSO",
"Dr02" : "Dyngo4a",
"Dr03" : "EIPA",
"Dr04" : "Pitstop2",
"Dr05" : "CPZ",
'''

# Channels
'''
"2obj" : ["_", "virus-all", "virus-extra", "nucleus"],
"3obj" : ["_", "virus-all", "EEA1", "nucleus"],
'''

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Paths
    # "data_path" : Path("D:\local_Suzuki\data"),
    "data_path" : Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data"),
    "tags"      : ["2OBJ"],
    
    # Parameters
    
    "data_x" : "cyt_edt_avg",
    "data_y" : "C1_avg",
    "tags_0" : ["020min", "Dr01", "N00"],
    "tags_1" : ["020min", "Dr05", "N00"],
    
    }

#%% Function(s) ---------------------------------------------------------------

def filter_data(df, name, tags):
    if tags:
        mask = df["path"].apply(lambda x: all(tag in x for tag in tags))
    else:
        mask = pd.Series(True, index=df.index)
    return df.loc[mask, name]

#%% Class(Plot) ---------------------------------------------------------------

class Plot:
    
    def __init__(self, parameters=parameters):
        
        # Fetch
        self.data_path = parameters["data_path"]
        self.tags      = parameters["tags"]
        self.data_x    = parameters["data_x"]
        self.data_y    = parameters["data_y"]
        self.tags_0    = parameters["tags_0"]
        self.tags_1    = parameters["tags_1"]
        
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
        x_m = filter_data(self.df_m, self.data_x, [])
        y_m = filter_data(self.df_m, self.data_y, [])
        x_v_m = filter_data(self.df_v_m, self.data_x, [])
        y_v_m = filter_data(self.df_v_m, self.data_y, [])
    
        # Initialize plot
        fig = plt.figure(figsize=(6, 9), layout="tight")
        gs = GridSpec(3, 2, figure=fig)
        
        # Info
        info = (   
            
            f"all n   : {len(x_m)}\n"
            f"valid n : {len(x_v_m)}\n"
            
            )
    
        # Scatter plot
        
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.scatter(x_m, y_m, label="all", s=5, alpha=0.5)
        ax0.scatter(x_v_m, y_v_m, label="valid", s=5, alpha=0.5)
        
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
        x0_v_m = filter_data(self.df_v_m, self.data_x, self.tags_0)
        y0_v_m = filter_data(self.df_v_m, self.data_y, self.tags_0)
        x1_v_m = filter_data(self.df_v_m, self.data_x, self.tags_1)
        y1_v_m = filter_data(self.df_v_m, self.data_y, self.tags_1)
        
        # Initialize plot
        fig = plt.figure(figsize=(6, 9), layout="tight")
        gs = GridSpec(3, 2, figure=fig)
        
        # Info
        info = (   
            
            f"cond0 n      : {len(x0_v_m)}\n"
            f"cond0 tag(s) : {','.join(self.tags_0)}\n"
            f"cond1 n      : {len(x1_v_m)}\n"
            f"cond1 tag(s) : {','.join(self.tags_1)}\n"
            
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
