#%% Imports -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

# matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Inputs --------------------------------------------------------------------

parameters = {
    
    # Paths
    # "data_path" : Path("D:\local_Suzuki\data"),
    "data_path" : Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data"),
    "exp"       : "2OBJ",
    
    # Parameters
    
    "data_x"    : "",
    "data_y"    : "",
    "tag0_in"   : [],
    "tag0_out"  : [],
    "tag1_in"   : [],
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
        self.exp       = parameters["exp"]
        self.data_x    = parameters["data_x"]
        self.data_y    = parameters["data_y"]
        self.tag0_in   = parameters["tag0_in"]
        self.tag0_out  = parameters["tag0_out"]
        self.tag1_in   = parameters["tag1_in"]
        self.tag1_out  = parameters["tag1_out"]
        
        # Run
        self.load()
        self.all_vs_valid()
        
#%% Class(Plot) : initialize() ------------------------------------------------

    def load(self):
        
        self.df_m = pd.read_csv(
            self.data_path / self.exp / "C2_results_m.csv")
        self.df_v_m = pd.read_csv(
            self.data_path / self.exp / "C2_results_v_m.csv")
        
#%% Class(Plot) : all_vs_valid() ----------------------------------------------

    def all_vs_valid(self):
        
        # Filter data
        x_m = filter_data(self.df_m, self.data_x, self.tag0_in, self.tag0_out)
        y_m = filter_data(self.df_m, self.data_y, self.tag0_in, self.tag0_out)
        x_v_m = filter_data(self.df_v_m, self.data_x, self.tag0_in, self.tag0_out)
        y_v_m = filter_data(self.df_v_m, self.data_y, self.tag0_in, self.tag0_out)
    
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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    plot = Plot()

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
    
    # def plot_valid(data0, data1, tag_in=[], tag_out=[]):
        
    #     # Load
    #     data_path = parameters["data_path"]
    #     df_m = pd.read_csv(data_path / "2OBJ" / "C2_results_m.csv")
    #     df_v = pd.read_csv(data_path / "2OBJ" / "C2_results_v_m.csv")
        
    #     # Filter data
    #     x0 = filter_data(df_m, data0, tag_in, tag_out)
    #     y0 = filter_data(df_m, data1, tag_in, tag_out)
    #     x1 = filter_data(df_v, data0, tag_in, tag_out)
    #     y1 = filter_data(df_v, data1, tag_in, tag_out)
    
    #     # Initialize plot
        
    #     fig = plt.figure(figsize=(6, 9), layout="tight")
    #     gs = GridSpec(3, 2, figure=fig)
    
    #     # Scatter plot
        
    #     ax0 = fig.add_subplot(gs[0, :2])
    #     ax0.scatter(x0, y0, label="all", s=5)
    #     ax0.scatter(x1, y1, label="valid", s=5)
        
    #     ax0.set_title(f"{data1} vs. {data0}")
    #     ax0.set_ylabel(data1)
    #     ax0.set_xlabel(data0)
    #     ax0.legend(loc="upper right")
        
    #     # Box plots
        
    #     boxplot_params = {"widths" : 0.6, "showfliers" : False}
        
    #     ax1 = fig.add_subplot(gs[1, 0])
    #     ax1.boxplot(y0, positions=[0], tick_labels=["all"], **boxplot_params)
    #     ax1.boxplot(y1, positions=[1], tick_labels=["valid"], **boxplot_params)
    #     ax1.set_title(data1)
    #     ax1.set_ylabel(data1)   
        
    #     ax2 = fig.add_subplot(gs[1, 1])
    #     ax2.boxplot(x0, positions=[0], tick_labels=["all"], **boxplot_params)
    #     ax2.boxplot(x1, positions=[1], tick_labels=["valid"], **boxplot_params)
    #     ax2.set_title(data0)
    #     ax2.set_ylabel(data0)   
        
    # plot_valid("cyt_edt_avg", "C3_avg", tag_in=[], tag_out=[])
        
    # def plot_conditions(
    #         data, tag0_in=[], tag0_out=[], tag1_in=[], tag1_out=[]):
        
    #     # Load
    #     data_path = parameters["data_path"]
    #     df_m = pd.read_csv(data_path / "2OBJ" /"C2_results_v_m.csv")
        
    #     # Filter data
    #     y0 = filter_data(df_m, data, tag0_in, tag0_out)
    #     y1 = filter_data(df_m, data, tag1_in, tag1_out)
        
    #     # Initialize plot
        
    #     fig = plt.figure(figsize=(6, 9), layout="tight")
    #     gs = GridSpec(3, 2, figure=fig)
    
    #     # Scatter plot
        
    #     ax0 = fig.add_subplot(gs[0, :2])
    #     ax0.scatter(x0, y0, label="valid", s=5)
        
    #     ax0.set_title(f"{data} vs. {data}")
    #     ax0.set_ylabel(data)
    #     ax0.set_xlabel(data)
    #     ax0.legend(loc="upper right")
        
    #     # Box plots
        
    #     boxplot_params = {"widths" : 0.6, "showfliers" : False}
        
    #     ax1 = fig.add_subplot(gs[1, 0])
    #     ax1.boxplot(y0, positions=[0], tick_labels=["all"], **boxplot_params)
    #     ax1.set_title(data1)
    #     ax1.set_ylabel(data1)   
        
    #     ax2 = fig.add_subplot(gs[1, 1])
    #     ax2.boxplot(x0, positions=[0], tick_labels=["all"], **boxplot_params)
    #     ax2.set_title(data0)
    #     ax2.set_ylabel(data0) 