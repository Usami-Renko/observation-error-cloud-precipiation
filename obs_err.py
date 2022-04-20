'''
Description: place observation error configuration data here
!!! Codes implemented here have to be rewritten in Fortran
Author: Hejun Xie
Date: 2022-04-18 19:41:05
LastEditors: Hejun Xie
LastEditTime: 2022-04-19 21:48:10
'''

import numpy as np

'''
Tropical boxes for bi-mode scaling
'''
Tropical_BOXES = [ \
    [ -10.,  20.,  90., 160.], 
    [   0.,  15., 160., 360.],
    ]

'''
Stats from warm start cycle [202107100900-202107251500]
Using bi-mode rescaling (Tropical and Extra-Tropical)
'''
obs_err_data = { \
    0: {'Cclr':0.000, 'Ccld':0.600, 'gclr':5.63,  'gcld':16.33, 'k':17.84},
    1: {'Cclr':0.000, 'Ccld':0.500, 'gclr':8.94,  'gcld':26.77, 'k':35.66},
    2: {'Cclr':0.000, 'Ccld':0.500, 'gclr':2.32,  'gcld':22.65, 'k':40.65},
    3: {'Cclr':0.000, 'Ccld':0.500, 'gclr':3.71,  'gcld':41.39, 'k':75.36},
    4: {'Cclr':0.000, 'Ccld':0.450, 'gclr':3.83,  'gcld':12.93, 'k':20.22},
    5: {'Cclr':0.000, 'Ccld':0.450, 'gclr':6.47,  'gcld':24.76, 'k':40.65},
    6: {'Cclr':0.000, 'Ccld':0.450, 'gclr':4.24,  'gcld':24.53, 'k':45.11},
    7: {'Cclr':0.000, 'Ccld':0.450, 'gclr':6.89,  'gcld':51.57, 'k':99.28},
    8: {'Cclr':0.000, 'Ccld':0.450, 'gclr':2.09,  'gcld':22.94, 'k':46.33},
    9: {'Cclr':0.000, 'Ccld':0.500, 'gclr':10.95, 'gcld':23.06, 'k':24.21},
}

def calc_obs_err(ch_no, c37):
    '''
    Calculate observation error from offline observation data
    '''
    k = obs_err_data[ch_no]['k']
    gclr = obs_err_data[ch_no]['gclr']
    gcld = obs_err_data[ch_no]['gcld']
    Cclr = obs_err_data[ch_no]['Cclr']
    Ccld = obs_err_data[ch_no]['Ccld']
    
    obs_err = np.full(c37.shape, np.nan, dtype=float)
    obs_err = np.where(c37<Cclr, gclr, obs_err)
    obs_err = np.where((c37>=Cclr) & (c37<=Ccld), gclr + k * (c37-Cclr), obs_err)
    obs_err = np.where(c37>Ccld, gcld, obs_err)
    return obs_err

'''
Simulation
'''
perc_segs_B = {
    'Global': \
    [+0.000000e+00, +5.000000e-01, +5.000000e+00, +6.000000e+01, +8.000000e+01,
     +9.000000e+01, +9.500000e+01, +9.900000e+01, +9.990000e+01, +1.000000e+02],
    'Tropical': \
    [+0.000000e+00, +5.000000e-01, +5.000000e+00, +6.000000e+01, +8.000000e+01,
     +9.000000e+01, +9.500000e+01, +9.900000e+01, +9.990000e+01, +1.000000e+02],
    'Extra-tropical': \
    [+0.000000e+00, +5.000000e-01, +5.000000e+00, +6.000000e+01, +8.000000e+01,
     +9.000000e+01, +9.500000e+01, +9.900000e+01, +9.990000e+01, +1.000000e+02],
}

value_segs_B = {
    'Global': \
    [-4.865806e-03, -2.313333e-04, +4.691338e-04, +1.948804e-03, +1.883265e-03,
     +7.384269e-03, +7.544142e-03, +2.836480e-02, +2.815900e-02, +7.292344e-02,
     +7.348882e-02, +1.331093e-01, +1.369136e-01, +4.319926e-01, +4.369338e-01,
     +7.296439e-01, +7.227672e-01, +8.820675e-01],
    'Tropical': \
    [-1.387933e-04, +2.728583e-03, +2.431249e-03, +3.270857e-03, +3.492306e-03,
     +9.478757e-03, +9.513192e-03, +2.549883e-02, +2.609100e-02, +6.809825e-02,
     +6.817824e-02, +1.450962e-01, +1.492198e-01, +4.272148e-01, +4.324713e-01,
     +7.087705e-01, +7.207985e-01, +9.059039e-01],
    'Extra-tropical': \
    [-5.105105e-03, -6.969258e-04, +1.266950e-04, +1.969512e-03, +1.825424e-03,
     +6.450837e-03, +6.633629e-03, +2.921423e-02, +2.885881e-02, +7.404979e-02,
     +7.489859e-02, +1.313197e-01, +1.367788e-01, +4.321832e-01, +4.388251e-01,
     +7.339082e-01, +7.241707e-01, +8.765834e-01],
}

bases_B = {
    'Global': \
    [+2.500000e-01, +2.750000e+00, +3.250000e+01, +7.000000e+01, +8.500000e+01,
     +9.250000e+01, +9.700000e+01, +9.945000e+01, +9.995000e+01],
    'Tropical': \
    [+2.500000e-01, +2.750000e+00, +3.250000e+01, +7.000000e+01, +8.500000e+01,
     +9.250000e+01, +9.700000e+01, +9.945000e+01, +9.995000e+01],
    'Extra-tropical': \
    [+2.500000e-01, +2.750000e+00, +3.250000e+01, +7.000000e+01, +8.500000e+01,
     +9.250000e+01, +9.700000e+01, +9.945000e+01, +9.995000e+01],
}

popts_B  = {
    'Global': \
    np.array([
            [+3.239345e+01, +2.861364e-04, -3.239599e+01], 
            [+1.494764e+01, +2.199783e-05, -1.494643e+01], 
            [+2.280179e-03, +3.709011e-02, +1.061026e-03], 
            [+6.269389e-03, +1.280619e-01, +5.802097e-03], 
            [+4.367332e-02, +9.847138e-02, +1.466558e-03], 
            [+5.222236e-02, +2.174614e-01, +4.316732e-02], 
            [+8.794027e-02, +6.447354e-01, +1.126934e-01], 
            [+1.642322e-01, +1.782827e+00, +3.633070e-01], 
            [-5.904840e+02, -2.697793e-03, +5.912864e+02]]),
    'Tropical': \
    np.array([
            [+1.651978e+01, +3.471446e-04, -1.651849e+01], 
            [+1.048957e+00, +1.778715e-04, -1.046107e+00], 
            [+3.363920e-03, +2.913702e-02, +1.982715e-03], 
            [+4.943670e-03, +1.257843e-01, +8.107872e-03], 
            [+2.796965e-02, +1.387804e-01, +1.211672e-02], 
            [+4.132361e-02, +3.325984e-01, +5.018605e-02], 
            [+1.950599e-01, +3.314747e-01, +4.869970e-02], 
            [+1.209144e-01, +2.174806e+00, +3.870301e-01], 
            [+2.638858e-01, +6.878212e+00, +5.337055e-01]]),
    'Extra-tropical': \
    np.array([
            [+2.535990e+01, +3.476496e-04, -2.536280e+01], 
            [+2.111207e+01, +1.939719e-05, -2.111102e+01], 
            [+1.542798e-03, +4.342645e-02, +1.358052e-03], 
            [+7.039905e-03, +1.250973e-01, +4.618624e-03], 
            [+4.838715e-02, +9.029563e-02, -1.948622e-03], 
            [+5.319937e-02, +2.032519e-01, +4.289280e-02], 
            [+6.745662e-02, +7.626716e-01, +1.221039e-01], 
            [+1.751123e-01, +1.701231e+00, +3.573845e-01], 
            [-6.505204e+02, -2.342934e-03, +6.513208e+02]]),
}

'''
Global Observation
'''
perc_segs_O = {
    'Global': \
    [+0.000000e+00, +2.500000e+00, +5.000000e+00, +1.000000e+01, +2.000000e+01,
     +4.000000e+01, +7.000000e+01, +9.000000e+01, +9.500000e+01, +1.000000e+02],
    'Tropical': \
    [+0.000000e+00, +2.500000e+00, +5.000000e+00, +1.000000e+01, +2.000000e+01,
     +4.000000e+01, +7.000000e+01, +9.000000e+01, +9.500000e+01, +1.000000e+02],
    'Extra-tropical': \
    [+0.000000e+00, +2.500000e+00, +5.000000e+00, +1.000000e+01, +2.000000e+01,
     +4.000000e+01, +7.000000e+01, +8.500000e+01, +9.200000e+01, +9.600000e+01,
     +9.900000e+01, +1.000000e+02],
}

value_segs_O = {
    'Global': \
    [-1.001873e-01, -9.960391e-02, -9.674651e-02, -8.114021e-02, -8.082279e-02,
     -6.342813e-02, -6.341753e-02, -4.369336e-02, -4.367288e-02, -1.625643e-02,
     -1.625119e-02, +3.119934e-02, +3.366442e-02, +1.316144e-01, +1.351066e-01,
     +2.683199e-01, +2.740738e-01, +9.455754e-01],
    'Tropical': \
    [-1.013679e-01, -9.681755e-02, -9.124297e-02, -7.141607e-02, -7.096391e-02,
     -5.032360e-02, -5.005578e-02, -2.488841e-02, -2.503213e-02, +9.866920e-03,
     +1.006880e-02, +7.930491e-02, +8.183379e-02, +3.016444e-01, +3.042611e-01,
     +5.249612e-01, +5.188872e-01, +9.554776e-01],
    'Extra-tropical': \
    [-1.000721e-01, -9.985047e-02, -9.770160e-02, -8.270866e-02, -8.245379e-02,
     -6.566230e-02, -6.554335e-02, -4.669038e-02, -4.663512e-02, -2.086878e-02,
     -2.098242e-02, +2.162277e-02, +2.206898e-02, +6.634289e-02, +6.718683e-02,
     +1.237318e-01, +1.252024e-01, +2.347710e-01, +2.370280e-01, +5.438090e-01,
     +5.517346e-01, +9.399780e-01],
}

bases_O = {
    'Global': \
    [+1.250000e+00, +3.750000e+00, +7.500000e+00, +1.500000e+01, +3.000000e+01,
     +5.500000e+01, +8.000000e+01, +9.250000e+01, +9.750000e+01],
    'Tropical': \
    [+1.250000e+00, +3.750000e+00, +7.500000e+00, +1.500000e+01, +3.000000e+01,
     +5.500000e+01, +8.000000e+01, +9.250000e+01, +9.750000e+01],
    'Extra-tropical': \
    [+1.250000e+00, +3.750000e+00, +7.500000e+00, +1.500000e+01, +3.000000e+01,
     +5.500000e+01, +7.750000e+01, +8.850000e+01, +9.400000e+01, +9.750000e+01,
     +9.950000e+01],
}

popts_O = {
    'Global': \
    np.array([
        [-4.297299e+00, -5.430047e-05, +4.197403e+00], 
        [+1.257453e+02, +4.964415e-05, -1.258343e+02], 
        [+2.253371e+02, +1.543879e-05, -2.254093e+02], 
        [+2.230037e+02, +8.844770e-06, -2.230573e+02], 
        [+4.087780e+02, +3.353464e-06, -4.088079e+02], 
        [+8.399604e-02, +1.858866e-02, -7.980835e-02], 
        [+4.471497e-02, +9.471622e-02, +1.632217e-02], 
        [+1.090837e-01, +2.311577e-01, +7.390208e-02], 
        [+2.641746e-01, +4.242436e-01, +1.826047e-01]]),
    'Tropical': \
    np.array([
        [-3.598389e+01, -5.058238e-05, +3.588480e+01], 
        [+1.427741e+02, +5.554761e-05, -1.428555e+02], 
        [+2.600408e+02, +1.587467e-05, -2.601015e+02], 
        [+2.765336e+02, +9.101017e-06, -2.765710e+02], 
        [+5.206488e+02, +3.351496e-06, -5.206564e+02], 
        [+7.348367e-02, +3.034756e-02, -3.654282e-02], 
        [+8.903314e-02, +1.037830e-01, +5.029625e-02], 
        [+2.148126e-01, +1.973740e-01, +1.731125e-01], 
        [+8.031725e-01, +1.074206e-01, -9.512689e-02]]),
    'Extra-tropical': \
    np.array([
        [-1.842703e+00, -4.811722e-05, +1.742741e+00], 
        [+1.401018e+02, +4.280585e-05, -1.401920e+02],
        [+1.671902e+02, +2.008669e-05, -1.672643e+02], 
        [+2.688676e+02, +7.011989e-06, -2.689237e+02], 
        [+3.728174e+02, +3.455624e-06, -3.728512e+02], 
        [+8.201925e-02, +1.712613e-02, -8.442036e-02],
        [+5.128328e-02, +5.590244e-02, -1.165120e-02], 
        [+4.737658e-02, +1.617272e-01, +4.028800e-02], 
        [+7.802657e-02, +3.272026e-01, +8.464804e-02], 
        [+1.864638e-01, +5.001607e-01, +1.489700e-01], 
        [+3.106719e-01, +1.180024e+00, +3.795227e-01]]),
}

def values2Percentiles(values, value_segs, popts, bases):
    '''
    The inverse of percentiles2Values
    There are gaps and intersections between value segments
    Therefore should be dealt with more carefully
    '''
    values = np.clip(np.array(values), value_segs[0], value_segs[-1]) # clip input
    percentiles = np.full(values.shape, np.nan)

    def inverse_exp(x, a, b, c, base):
        return np.log(((x-c) / a)) / b + base
    
    # if value lies in valid range of segmented function
    for irange in range(len(value_segs)//2):
        Range = (value_segs[irange*2], value_segs[irange*2+1])
        a, b, c = popts[irange]
        base = bases[irange]
        percentiles = np.where((values>=Range[0]) & (values<=Range[1]), \
            inverse_exp(values, a, b, c, base), percentiles)
    
    # if value lies in gap of segmented function
    for igap in range(len(value_segs)//2 - 1):
        Gap = (value_segs[igap*2+1], value_segs[igap*2+2])
        percentiles = np.where((values>=Gap[0]) & (values<=Gap[1]), \
            (Gap[0]+Gap[1])/2., percentiles)

    percentiles = np.clip(percentiles, 0., 100.) # clip output
    return percentiles
        
def percentiles2Values(percentiles, perc_segs, popts, bases):
    percentiles = np.clip(np.array(percentiles), perc_segs[0], perc_segs[-1]) # clip input
    values = np.full(percentiles.shape, np.nan)

    def exp(x, a, b, c, base):
        return a * np.exp(b * (x - base)) + c
        
    for irange in range(len(perc_segs)-1):
        Range = (perc_segs[irange], perc_segs[irange+1])
        a, b, c = popts[irange]
        base = bases[irange]
        values = np.where((percentiles>=Range[0]) & (percentiles<=Range[1]), \
            exp(percentiles, a, b, c, base), values)
    
    values = np.clip(values, -0.1, 1.0) # clip output

    return values


def rescale_B2O(values_B, region='Global'):
    '''
    Wrapper for rescale_B2O_params
    '''    
    return rescale_B2O_params(values_B, 
            value_segs_B[region], popts_B[region], bases_B[region], 
            perc_segs_O[region], popts_O[region], bases_O[region])
    
def rescale_B2O_params(values_B, \
                       value_segs_B, popts_B, bases_B, \
                       perc_segs_O, popts_O, bases_O):
    percs    = values2Percentiles(values_B, value_segs_B, popts_B, bases_B)
    values_O = percentiles2Values(percs   ,  perc_segs_O, popts_O, bases_O)

    return values_O 

if __name__ == "__main__":
    ch_no = 8
    c37 = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 0.6, 0.8, 1.0, 1.1])
    print(calc_obs_err(ch_no, c37))
    