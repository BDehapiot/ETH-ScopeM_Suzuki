#%% Imports -------------------------------------------------------------------

import numpy as np
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
       
    "rmt_path" : 
        Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data\\2OBJ"),
    "loc_path" : 
        Path("D:\local_Suzuki\data\\2OBJ"),
        
    "save" : "loc",
    
    # Parameters
    
    "data_x" : "cyt_edt_avg",
    "data_y" : "C1_msk_avg",
    "tags_0" : ["000min", "IM00"],
    "tags_1" : ["020min", "IM00"],
    
    # Statistics
    "error" : "sem",
    
    }

#%% Function(s) ---------------------------------------------------------------

def filter_data(df, name, tags):
    if tags:
        mask = df["path"].apply(lambda x: all(tag in x for tag in tags))
    else:
        mask = pd.Series(True, index=df.index)
    return df.loc[mask, name]

def stats_data(data, err="sem"):
    avg = np.mean(data)
    if err == "std":
        err = np.std(data)
    elif err == "sem":
        err = np.std(data, ddof=1) / np.sqrt(len(data))
    return avg, err   

#%% Class(Plot) ---------------------------------------------------------------

class Plot:
    
    def __init__(self, parameters=parameters):
        
        # Fetch
        self.parameters = parameters
        self.save = self.parameters["save"]
        self.error = self.parameters["error"]
        self.data_x = parameters["data_x"]
        self.data_y = parameters["data_y"]
        self.tags_0 = parameters["tags_0"]
        self.tags_1 = parameters["tags_1"]
        
        # Run
        self.load()
        # self.all_vs_valid()
        self.cond0_vs_cond1()
       
#%% Class(Plot) : load() ------------------------------------------------------

    def load(self):

        self.df_m = pd.read_csv(
            self.parameters[f"{self.save}_path"] / "C2_results_m.csv")
        self.df_v_m = pd.read_csv(
            self.parameters[f"{self.save}_path"] / "C2_results_v_m.csv")
        
#%% Class(Plot) : all_vs_valid() ----------------------------------------------

    def all_vs_valid(self):
        
        # Filter data
        x_m = filter_data(self.df_m, self.data_x, [])
        y_m = filter_data(self.df_m, self.data_y, [])
        x_v_m = filter_data(self.df_v_m, self.data_x, [])
        y_v_m = filter_data(self.df_v_m, self.data_y, [])
        
        # Statistics
        x_m_avg, x_m_err = stats_data(x_m, err=self.error)
        y_m_avg, y_m_err = stats_data(y_m, err=self.error)
        x_v_m_avg, x_v_m_err = stats_data(x_v_m, err=self.error)
        y_v_m_avg, y_v_m_err = stats_data(y_v_m, err=self.error)
    
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
        
        # Bar plots
        
        barplot_params = {"width" : 0.75, "capsize" : 10}
        
        ax1 = fig.add_subplot(gs[1, 0]) 
        ax1.bar(0, y_m_avg, yerr=y_m_err, **barplot_params)
        ax1.bar(1, y_v_m_avg, yerr=y_v_m_err, **barplot_params)
        ax1.set_title(f"{self.data_y} ({self.error})")
        ax1.set_ylabel(self.data_y)  
        ax1.set_xticks([0, 1], ["all", "valid"])
        
        ax2 = fig.add_subplot(gs[1, 1]) 
        ax2.bar(0, x_m_avg, yerr=x_m_err, **barplot_params)
        ax2.bar(1, x_v_m_avg, yerr=x_v_m_err, **barplot_params)
        ax2.set_title(f"{self.data_x} ({self.error})")
        ax2.set_ylabel(self.data_x)  
        ax2.set_xticks([0, 1], ["all", "valid"])

#%% Class(Plot) : cond0_vs_cond1() --------------------------------------------

    def cond0_vs_cond1(self):
        
        # Filter data
        x0_v_m = filter_data(self.df_v_m, self.data_x, self.tags_0)
        y0_v_m = filter_data(self.df_v_m, self.data_y, self.tags_0)
        x1_v_m = filter_data(self.df_v_m, self.data_x, self.tags_1)
        y1_v_m = filter_data(self.df_v_m, self.data_y, self.tags_1)
        
        # Statistics
        x0_v_m_avg, x0_v_m_err = stats_data(x0_v_m, err=self.error)
        y0_v_m_avg, y0_v_m_err = stats_data(y0_v_m, err=self.error)
        x1_v_m_avg, x1_v_m_err = stats_data(x1_v_m, err=self.error)
        y1_v_m_avg, y1_v_m_err = stats_data(y1_v_m, err=self.error)
        
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
                
        # Bar plots
        
        barplot_params = {"width" : 0.75, "capsize" : 10}
        
        ax1 = fig.add_subplot(gs[1, 0]) 
        ax1.bar(0, y0_v_m_avg, yerr=y0_v_m_err, **barplot_params)
        ax1.bar(1, y1_v_m_avg, yerr=y1_v_m_err, **barplot_params)
        ax1.set_title(f"{self.data_y} ({self.error})")
        ax1.set_ylabel(self.data_y)  
        ax1.set_xticks(
            [0, 1], [f"{'-'.join(self.tags_0)}", f"\n{'-'.join(self.tags_1)}"])
        
        ax2 = fig.add_subplot(gs[1, 1]) 
        ax2.bar(0, x0_v_m_avg, yerr=x0_v_m_err, **barplot_params)
        ax2.bar(1, x1_v_m_avg, yerr=x1_v_m_err, **barplot_params)
        ax2.set_title(f"{self.data_x} ({self.error})")
        ax2.set_ylabel(self.data_x)  
        ax2.set_xticks(
            [0, 1], [f"{'-'.join(self.tags_0)}", f"\n{'-'.join(self.tags_1)}"])

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    plot = Plot()
    df_m = plot.df_m
    df_v_m = plot.df_v_m