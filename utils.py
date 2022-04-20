'''
Description: place small utils here
Author: Hejun Xie
Date: 2022-04-20 19:04:28
LastEditors: Hejun Xie
LastEditTime: 2022-04-20 21:22:18
'''
import numpy as np


def chkey(ch_no):
    '''
    ch_no: 0, 1, 2, ......
    '''
    return 'ch{}'.format(ch_no+1)

def get_channel_str(instrument, ch_no):
    mwri_str = {
        0: '10.65GHz V pol.',
        1: '10.65GHz H pol.',
        2: '18.7GHz V pol.',
        3: '18.7GHz H pol.',
        4: '23.8GHz V pol.',
        5: '23.8GHz H pol.',
        6: '36.5GHz V pol.',
        7: '36.5GHz H pol.',
        8: '89.0GHz V pol.',
        9: '89.0GHz H pol.',
    }

    if instrument == 'mwri':
        return 'FY3D MWRI ' + mwri_str[ch_no]

def get_slice(x, xrange):
    '''
    Return x index slice in xrange [xmin, max]
    '''
    x_idx_min = np.argmin(np.abs(x - xrange[0]))
    x_idx_max = np.argmin(np.abs(x - xrange[1]))
    return slice(x_idx_min, x_idx_max+1)

def gaussian(x, mu, sigma, sumy):
    '''
    Return a guassian distribution on x
    with mean=mu, and std=sigma
    and intergrates in the range of x to sumy
    '''
    gaussian_y =  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    sum_unit = np.sum(gaussian_y)
    gaussian_scale = gaussian_y * (sumy / sum_unit)
    return gaussian_scale

def get_vminvmax(data, vstage=None, ign_perc=0., lsymmetric=False):
    '''
    Description:
        Get vmax, vmin to plot with.
    Params:
        data: valid value: A single numpy.ndarray or a list of numpy.ndarray.
                The data to plot with, or list of data to plot and share the same vmin / vmax
        vstage: valid value: An integer or a real number or None.
                The varying stage of vmin / vmax, to preserve reasonable tick levels.
                if vstage is None, then no stage is performed
        ign_perc: valid value: An integer or a real number in [0., 50.].
                The ignore percent to avoid singularity values spoiling the vmin / vmax
                percentile: [ign_perc, 100.-ign_perc].
        lsymmetric: Valid value: True or False. 
                If the wanted vmin and vmax is symmetric with respect to 0.
    Returns:
        vmin, max
    '''

    import numpy as np
    from math import ceil, floor

    '''
    0. Make list
    '''
    if not isinstance(data, list):
        data = [data]
    
    '''
    1. Remove np.nan values
    '''
    valid_data = []
    for idata in data:
        imask = np.isnan(idata)
        valid_data.append(idata[~imask])

    '''
    2. Get MIN, MAX with numpy.percentile
    '''
    MIN = min([np.percentile(idata, ign_perc) for idata in valid_data])
    MAX = max([np.percentile(idata, 100.-ign_perc) for idata in valid_data])

    '''
    3. Add stage on MIN, MAX 
    '''
    if vstage:
        MIN = floor(MIN / vstage) * vstage
        MAX = ceil(MAX / vstage) * vstage

    '''
    4. make it symmetric if wanted
    '''
    if lsymmetric:
        MAX = max(np.abs(MIN), np.abs(MAX))
        MIN = - MAX

    return MIN, MAX

def cornerTag(ax, text, fontsize):
    ax.text(1.0, 1.0, text, ha='right', va='bottom', size=fontsize, transform=ax.transAxes)