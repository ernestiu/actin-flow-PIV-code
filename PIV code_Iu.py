#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:44:52 2023

A part of this code is adopted from: 
    https://openpiv.readthedocs.io/en/latest/src/tutorial1.html


@author: ernestiu
"""

from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import pathlib
import cv2
import io as io_2
import tifffile
import pandas as pd
from skimage.measure import find_contours

def display_vector_field(
    filename,
    image_name = None,
    window_size = None,
    show_invalid = True,
    scale = None,
    **kw):

    a = np.loadtxt(filename)
    
    x, y, u, v, flags, mask = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    fig = ax.get_figure()

    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.
    
    # now mark the valid/invalid vectors
    invalid = flags > 0 # mask.astype("bool")  
    valid = ~invalid
    
    hypotenuse = np.hypot(u[valid], v[valid])*60 # get the hypotenuse and convert it from second to min
    
    quiver_plot = ax.quiver(x[valid],y[valid],u[valid],v[valid], hypotenuse, pivot='middle',
        headwidth=2, headlength=5, width=0.008, #0.008 # scale here controls the size of the arrows; a smaller num gives a bigger arrow
        scale=0.2, #0.0015
        scale_units='inches',
        clim=[0,15],
        cmap='viridis_r')
    
    cb = fig.colorbar(quiver_plot, orientation='vertical')
    cb.set_label('Velocity (µm/min)', fontsize=20)
    cb.ax.tick_params(labelsize=18) 
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return fig, ax

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=300):
    buf = io_2.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # for cv2 videowriter, you keep BGR (not RGB)
    return img

def cart2pol(x, y):
    '''
    This code takes the u and v arrays and returns its polar coordinates.
    ----------
    x : array
        u component
    y : array
        v component

    Returns
    -------
    None.

    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) * 180/np.pi # convert from radian to degree
    phi %= 360 # convert all negative degrees to positive
    return rho, phi



image_path = 'image.tif'

sub_sample = 5

dt = 3*sub_sample # sec, time interval between the two frames

fig_width, fig_height = 7, 7 # inch
pixel_size = 0.0733333 # micron per pixel


winsize = 30  # pixels, interrogation window size in frame A
searchsize = 30  # pixels, search area size in frame B
overlap = 0 # pixels, 50% overlap


image_name = os.path.splitext(os.path.basename(image_path))[0]

img = io.imread(image_path)

img = img[::sub_sample]

piv_img = np.empty(img.shape)

img_array = []
piv_directory = os.path.dirname(image_path) + '/' + image_name 

all_mean = [] 
all_median = [] 
all_std = []
all_max = [] 
all_min = [] 
all_phi = []
all_hypotenuse = []


for frame in range(img.shape[0]-1):
    frame_a  = img[frame,:,:]
    frame_b  = img[frame+1,:,:]
        
    
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32),
        window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak')
    

    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=searchsize, overlap=overlap)

    
    invalid_mask = validation.sig2noise_val(sig2noise,threshold = 1)
    
    u2, v2 = filters.replace_outliers(u0, v0, invalid_mask, method='localmean', max_iter=3, kernel_size=3)
    
    # scaling_factor is how many pixels per micron
    scaling_factor = 1/pixel_size 
    # convert x,y to µm and u,v to µm/sec
    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = scaling_factor)      
    
    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
    
    hypotenuse = np.hypot(u3, v3) * 60 # convert from sec to min
    all_hypotenuse.append(hypotenuse)


    rho, phi = cart2pol(u3, v3)
    all_phi.append(phi)


    # the use of nan is to ignore nan in the calculations
    all_mean.append(np.nanmean(hypotenuse))
    all_median.append(np.nanmedian(hypotenuse))
    all_std.append(np.nanstd(hypotenuse))
    all_max.append(np.nanmax(hypotenuse))
    all_min.append(np.nanmin(hypotenuse))
    
    
    if not os.path.exists(piv_directory):
        # If it doesn't exist, create it
        os.makedirs(piv_directory)
    save_to_this_path = piv_directory + '/' + image_name + '_PIV_' + str(frame+1) + '.txt'
    tools.save(save_to_this_path , x, y, u3, v3, invalid_mask)
    
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    
    xmax = np.amax(x) + winsize / (2 * scaling_factor)
    ymax = np.amax(y) + winsize / (2 * scaling_factor)
    
    ax.imshow(frame_a, zorder=0, cmap='gray_r', extent=[0.0, xmax, 0.0, ymax])

    display_vector_field(
        pathlib.Path(save_to_this_path),
        show_invalid = False)
    
    # # you can get a high-resolution image as numpy array!!
    piv_frame = get_img_from_fig(fig)
    # Check if the directory exists
    piv_directory = os.path.dirname(image_path) + '/' + image_name 
    

    img_array.append(piv_frame)
    height, width, layers = piv_frame.shape
    size = (width,height)
    
    print('Progress: {}/{}'.format(str(frame+1), img.shape[0]-1))

out = cv2.VideoWriter(os.path.dirname(image_path) +'/' + image_name + '_piv.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 7, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

df = pd.DataFrame()
df['Mean'] = pd.Series(np.mean(all_mean))
df['Median'] = pd.Series(np.mean(all_median))
df['Std'] = pd.Series(np.mean(all_std))
df['Max'] = pd.Series(np.mean(all_max))
df['Minimum'] = pd.Series(np.mean(all_min))
# df.to_excel(os.path.dirname(image_path) +'/' + image_name + '_data.xlsx', engine='xlsxwriter', index=False)



all_first_row = []
all_second_row = []
all_the_rest = []
for n in range(len(all_phi)):
    all_first_row.extend(all_phi[n][0])
    all_second_row.extend(all_phi[n][1])
    all_the_rest.extend(all_phi[n][2::].ravel())



df2 = pd.DataFrame(all_phi[0])
df3_a = pd.DataFrame()
df3_a['First row'] = pd.Series(all_first_row) #all_phi[0][0]
df3_a['Second row'] = pd.Series(all_second_row) #all_phi[0][1]
df3_b = pd.DataFrame()
df3_b['The rest'] = pd.Series(all_the_rest)
df3 = pd.concat([df3_a, df3_b], ignore_index=True)
df4 = pd.DataFrame(all_hypotenuse[0])


writer = pd.ExcelWriter(os.path.dirname(image_path) +'/' + image_name + '_data.xlsx', engine = 'xlsxwriter')
df.to_excel(writer, sheet_name = 'PIV summary', index=False)
df2.to_excel(writer, sheet_name = 'PIV angles', index=False)
df3.to_excel(writer, sheet_name = 'PIV angles_transposed', index=False)
df4.to_excel(writer, sheet_name = 'PIV speed', index=False)
writer.close()

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=False, edge_color='C0'):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = np.radians(x)# * np.pi/180 # convert from degrees to radians
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    
    radius = np.array([i/sum(n)*100 for i in n]) # normalize to %
    
    # Compute width of each bin
    widths = np.diff(bins)


    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=2, align='edge', width=widths,
                     color=edge_color,edgecolor=edge_color, fill=True, linewidth=2, alpha=0.8)


    return n, bins, patches

# Construct figure and axis to plot on
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), dpi=300)
# fig, ax = plt.subplots(1, 1, dpi=300)

colors = {'First row':'#003f5c', 'Second row':'#bc5090', 'The rest':'#ffa600'}  
# Visualise by area of bins
circular_hist(ax, df3['The rest'].dropna(), density=True, edge_color='#ffa600')
circular_hist(ax, df3['Second row'].dropna(), density=True, edge_color='#bc5090')
circular_hist(ax, df3['First row'].dropna(), density=True, edge_color='#003f5c') #


labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
fig.legend(handles, labels)

fig.savefig(os.path.dirname(image_path) +'/' + image_name + '_polar_hist.pdf')
plt.show()

from skimage.transform import rescale, resize, downscale_local_mean


PIV_img = np.array(all_hypotenuse[0])
                   
scaling_factor = img[0].shape[0]/PIV_img.shape[0]

image_rescaled = rescale(PIV_img, scaling_factor, anti_aliasing=False)

fig_2, ax_2 = plt.subplots(1, 1, dpi=300)
ax_2.imshow(img[0], cmap='gray_r')
ax_2.imshow(image_rescaled, alpha=0.4, cmap='jet')

plt.axis('off')
plt.tight_layout()
fig_2.savefig(os.path.dirname(image_path) +'/' + image_name + '_PIV_heatmap.pdf')
plt.show()

       


