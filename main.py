#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script generates test case data and 2-D plotting of 3-way tuples with 
# ternary plots


#%%============================================================================
# import
# =============================================================================

# import sys
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import ternary
import regionmask as rm
import geopandas as gp
from copy import deepcopy as dp
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as feature
import itertools

from ternary.helpers import unzip, normalize, simplex_iterator, permute_point, project_point
from ternary.colormapping import get_cmap, colormapper

#%%============================================================================
# functions
# =============================================================================

def myround(x, base=5):
    return base * round(x/base)

def f(x):
    return (0.4 * x[0] + 0.1 * x[1]) / (0.5 * x[0] + 0.5 * x[1] + 0.5 * x[2])

def fx(x1,x2,x3):
    return (0.4 * x1 + 0.1 * x2) / (0.5 * x1 + 0.5 * x2 + 0.5 * x3)

def converter(ds):
    x1 = ds['arr1']
    x2 = ds['arr2']
    x3 = ds['arr3']
    x = [x1,x2,x3]
    arr4 = xr.apply_ufunc(
        fx, # function 
        x1,x2,x3, # 3 data arrays as input
        input_core_dims=[[],[],[]],
        vectorize=True,
    )
    return arr4

#%%============================================================================
# path
#==============================================================================

curDIR = r'C:/Users/lgrant/Documents/repos/ternary'
sfDIR = os.path.join(curDIR, 'shapefiles')
os.chdir(curDIR)

# data input directories
outDIR = os.path.join(curDIR, 'figures')


#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_svplt=1      # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_analysis=0   # 0: 
                  # 1: 
                  
continents = {}
continents['North America'] = 7
continents['South America'] = 4
continents['Europe'] = 1
continents['Asia'] = 6
continents['Africa'] = 3
continents['Australia'] = 5

continent_names = []
for c in continents.keys():
    continent_names.append(c)

ns = 0
for c in continents.keys():
    ns += 1                      

# continental da
lon = np.arange(-179.5, 180, 0.5)
lat = np.arange(-89.5, 90, 0.5)
os.chdir(sfDIR)
gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
cnt_regs = rm.mask_geopandas(gpd_continents,lon,lat)
# cnt_regs = cnt_regs.where((ar6_regs != 0)&(ar6_regs != 20)&(ar6_land == 1))

# random data; seeds keep randoms same per session
cnt_dt = {}
for i,n in enumerate(continent_names):
    cnt_dt[n] = {}
    np.random.seed(i); cnt_dt[n]['arr1'] = myround(np.random.uniform(low=0,high=1)*100,base=10)
    np.random.seed(i); cnt_dt[n]['arr2'] = myround(np.random.uniform(low=0,high=(100-cnt_dt[n]['arr1'])),base=10)
    cnt_dt[n]['arr3'] = 100 -  cnt_dt[n]['arr1'] - cnt_dt[n]['arr2']

ds = xr.full_like(cnt_regs,0).to_dataset()
for arr in ['arr1','arr2','arr3']:
    data = dp(cnt_regs)
    for n in continent_names:
        data = xr.where(cnt_regs == continents[n],cnt_dt[n][arr],data)
    ds[arr] = data
ds = ds.drop('region')
    
arr4 = converter(ds)

#%%============================================================================
# plot
#==============================================================================

cstlin_lw = 0.75
extent = [-180,180,-60,90]
scale = 10
tick_multiple = 2
grid_multiple = 1
fig = plt.figure(figsize=(15,8))

# ax for map of virtual data
gs1 = gridspec.GridSpec(1,1)
m_left = 0.0
m_bottom = 0.0
m_right = 0.7
m_top = 1.0
m_rect = [m_left, m_bottom, m_right, m_top]
ax1 = fig.add_subplot(
    gs1[0],
    projection=ccrs.PlateCarree()
    )    
gs1.tight_layout(
    figure=fig,
    rect=m_rect,
    h_pad=2)

# ax for ternary colorbar
gs2 = gridspec.GridSpec(1,1)
t_left = 0.7
t_bottom = 0.25
t_right = 1.0
t_top = 0.75
t_rect = [t_left, t_bottom, t_right, t_top]
ax2 = fig.add_subplot(gs2[0])
gs2.tight_layout(
    figure=fig,
    rect=t_rect,
    h_pad=2
    )

# plot map stuff
arr4.plot(
    ax=ax1,
    cmap='terrain',
    vmin=0.0,
    vmax=1.0,
    add_colorbar=False,   
    )
ax1.coastlines(linewidth=cstlin_lw)
ax1.add_feature(
    feature.OCEAN,
    facecolor='lightsteelblue'
    )
ax1.set_extent(
    extent, 
    crs=ccrs.PlateCarree()
    )

# plot ternary
ax2.axis("off")
figure, tax = ternary.figure(ax=ax2, scale=scale)
axes_colors = {
    'b': 'dimgrey',
    'l': 'dimgrey',
    'r': 'dimgrey'
    }
tax.heatmapf(
    f, 
    boundary=True,
    colorbar=False,
    style="hexagonal", cmap=plt.cm.get_cmap('terrain'),
    cbarlabel='Component 0 uptake',
    vmax=1.0, vmin=0.0
    )
tax.boundary(
    linewidth=2.0,
    axes_colors=axes_colors
    )
tax.left_axis_label(
    "$x_1$",
    offset=0.16,
    color='k'
    )
tax.right_axis_label(
    "$x_0$",
    offset=0.16,
    color='k'
    )
tax.bottom_axis_label(
    "$x_2$",
    offset=0.06,
    color='k'
    )
tax.gridlines(
    multiple=grid_multiple,
    linewidth=1,
    horizontal_kwargs={'color': axes_colors['l']},
    left_kwargs={'color': axes_colors['r']},
    right_kwargs={'color': axes_colors['b']},
    alpha=0.7
    )
ticks = [i / float(scale) for i in range(scale+1)]
tax.ticks(
    ticks = [i / float(scale) for i in range(scale+1)],
    axis='rlb',
    # multiple=tick_multiple,
    axes_colors=axes_colors,
    offset=0.03,
    tick_formats = "%0.1f",
    # tick_formats="%i",
    clockwise=True
    )
tax.clear_matplotlib_ticks()
tax._redraw_labels()

# proofs with virtual data:
bbox = ax1.get_position()# [[xmin, ymin], [xmax, ymax]]
x_min = bbox.x0
x_max = bbox.x1
y_min = bbox.y0
y_max = bbox.y1    
# step =
for n in continent_names:
    
    if n == 'North America' or n == 'South America':
        if n == 'North America':
            y_minf = y_min
        else:
            y_minf = y_min - 0.1
        ax1.text(
            x_min,
            y_minf - 0.3,
            '{}; $x_0$: {}, $x_1$: {}, $x_2$: {},'.format(n,cnt_dt[n]['arr1'],cnt_dt[n]['arr2'],cnt_dt[n]['arr3']),
            transform=ax1.transAxes,
            fontsize=12
        )
    elif n == 'Europe' or n == 'Africa':
        if n == 'Europe':
            y_minf = y_min
        else:
            y_minf = y_min - 0.1        
        ax1.text(
            x_min + (x_max - x_min)/2,
            y_minf - 0.3,
            '{}; $x_0$: {}, $x_1$: {}, $x_2$: {},'.format(n,cnt_dt[n]['arr1'],cnt_dt[n]['arr2'],cnt_dt[n]['arr3']),
            transform=ax1.transAxes,
            fontsize=12
        )        
    elif n == 'Asia' or n == 'Australia':
        if n == 'Asia':
            y_minf = y_min
        else:
            y_minf = y_min - 0.1                
        ax1.text(
            x_min + (x_max - x_min),
            y_minf - 0.3,
            '{}; $x_0$: {}, $x_1$: {}, $x_2$: {},'.format(n,cnt_dt[n]['arr1'],cnt_dt[n]['arr2'],cnt_dt[n]['arr3']),
            transform=ax1.transAxes,
            fontsize=12
        )        
            
    
plt.tight_layout()
if flag_svplt == 1:
    plt.savefig(outDIR+'/ternary_color_mapping.png',dpi=200)


# # %%

# # Function to visualize for heat map
# def f(x):
#     return 1.0 * x[0] / (1.0 * x[0] + 0.2 * x[1] + 0.05 * x[2])

# # Dictionary of axes colors for bottom (b), left (l), right (r).
# axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}

# scale = 10

# fig, ax = plt.subplots()
# ax.axis("off")
# figure, tax = ternary.figure(ax=ax, scale=scale)

# tax.heatmapf(f, boundary=False,
#              style="hexagonal", cmap=plt.cm.get_cmap('Blues'),
#              cbarlabel='Component 0 uptake',
#              vmax=1.0, vmin=0.0)

# tax.boundary(linewidth=2.0, axes_colors=axes_colors)

# tax.left_axis_label("$x_1$", offset=0.16, color=axes_colors['l'])
# tax.right_axis_label("$x_0$", offset=0.16, color=axes_colors['r'])
# tax.bottom_axis_label("$x_2$", offset=0.06, color=axes_colors['b'])

# tax.gridlines(multiple=1, linewidth=2,
#               horizontal_kwargs={'color': axes_colors['b']},
#               left_kwargs={'color': axes_colors['l']},
#               right_kwargs={'color': axes_colors['r']},
#               alpha=0.7)

# # Set and format axes ticks.
# ticks = [i / float(scale) for i in range(scale+1)]
# tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
#           axes_colors=axes_colors, offset=0.03, tick_formats="%0.1f")

# tax.clear_matplotlib_ticks()
# tax._redraw_labels()
# plt.tight_layout()
# tax.show()




# %%
