'''
Description: place observation error configuration data here
!!! Codes implemented here have to be rewritten in Fortran
Author: Hejun Xie
Date: 2022-04-18 19:41:05
LastEditors: Hejun Xie
LastEditTime: 2022-05-13 20:24:11
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
obs_err_data = {}

'''
rescaling coefficients
'''
perc_segs_B = {} 
value_segs_B = {}
bases_B = {}
popts_B = {}
perc_segs_O = {}
value_segs_O = {}
bases_O = {}
popts_O = {}

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

def read_obs_err_data(infile):
    with open(infile, 'r') as fin:
        fin.readline()
        fin.readline()
        nchannels = int(fin.readline())
        fin.readline()
        for ichannel in range(nchannels):
            obs_err_data[ichannel] = dict()
            datapiece = [float(data) for data in fin.readline().split()]
            obs_err_data[ichannel]['Cclr'] = datapiece[1]
            obs_err_data[ichannel]['Ccld'] = datapiece[2]
            obs_err_data[ichannel]['gclr'] = datapiece[3]
            obs_err_data[ichannel]['gcld'] = datapiece[4]
            obs_err_data[ichannel]['k']    = datapiece[5]

read_obs_err_data('./cldraderr_mwi.dat')

def write_rescalingCoefficient(outfile, \
    perc_segs_B, value_segs_B, bases_B, popts_B, \
    perc_segs_O, value_segs_O, bases_O, popts_O):
    
    def get_1darray_format(nvalues):
        ncolumns = 5
        singleformat = '{:<+14.6e}'
        nlines = nvalues // ncolumns
        modcolumns = nvalues % ncolumns
        formatstr = (ncolumns * singleformat + '\n') * nlines + \
            (modcolumns * singleformat + '\n' if modcolumns else '')
        return formatstr

    with open(outfile, 'w') as fout:
        fout.write('! Rescaling Coefficient Data\n')
        fout.write('! 1. Simulation\n')
        fout.write('! nsegs\n')
        nsegs = len(bases_B)
        fout.write('{}\n'.format(nsegs))
        fout.write('! perc_segs\n')
        fout.write(get_1darray_format(nsegs+1).format(*perc_segs_B))
        fout.write('! value_segs\n')
        fout.write(get_1darray_format(2*nsegs).format(*value_segs_B))
        fout.write('! bases\n')
        fout.write(get_1darray_format(nsegs).format(*bases_B))
        fout.write('! a\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[0] for popt in popts_B]))
        fout.write('! b\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[1] for popt in popts_B]))
        fout.write('! c\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[2] for popt in popts_B]))

        fout.write('! 1. Observation\n')
        fout.write('! nsegs\n')
        nsegs = len(bases_O)
        fout.write('{}\n'.format(nsegs))
        fout.write('! perc_segs\n')
        fout.write(get_1darray_format(nsegs+1).format(*perc_segs_O))
        fout.write('! value_segs\n')
        fout.write(get_1darray_format(2*nsegs).format(*value_segs_O))
        fout.write('! bases\n')
        fout.write(get_1darray_format(nsegs).format(*bases_O))
        fout.write('! a\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[0] for popt in popts_O]))
        fout.write('! b\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[1] for popt in popts_O]))
        fout.write('! c\n')
        fout.write(get_1darray_format(nsegs).format(*[popt[2] for popt in popts_O]))

def read_rescalingCoefficient(infile):

    def read_1darray(fhandle, array):
        ncolumns = 5
        nvalues = array.shape[0]
        nlines = nvalues // ncolumns
        modcolumns = nvalues % ncolumns
        for iline in range(nlines):
            datapiece = np.array([float(data) for data in fhandle.readline().split()])
            array[iline*ncolumns:(iline+1)*ncolumns] = datapiece
        if modcolumns != 0:
            datapiece = np.array([float(data) for data in fhandle.readline().split()])
            array[-modcolumns:] = datapiece

    with open(infile, 'r') as fin:
        fin.readline()
        fin.readline()
        fin.readline()
        nsegs_B0 = int(fin.readline())
        perc_segs_B0 = np.zeros((nsegs_B0+1), dtype='float32')
        value_segs_B0 = np.zeros((2*nsegs_B0), dtype='float32')
        bases_B0 = np.zeros((nsegs_B0), dtype='float32')
        popts_B0 = np.zeros((nsegs_B0,3), dtype='float32')
        fin.readline()
        read_1darray(fin, perc_segs_B0)
        fin.readline()
        read_1darray(fin, value_segs_B0)
        fin.readline()
        read_1darray(fin, bases_B0)
        fin.readline()
        read_1darray(fin, popts_B0[:,0]) # a
        fin.readline()
        read_1darray(fin, popts_B0[:,1]) # b
        fin.readline()
        read_1darray(fin, popts_B0[:,2]) # c

        fin.readline()
        fin.readline()
        nsegs_O0 = int(fin.readline())
        perc_segs_O0 = np.zeros((nsegs_O0+1), dtype='float32')
        value_segs_O0 = np.zeros((2*nsegs_O0), dtype='float32')
        bases_O0 = np.zeros((nsegs_O0), dtype='float32')
        popts_O0 = np.zeros((nsegs_O0,3), dtype='float32')
        fin.readline()
        read_1darray(fin, perc_segs_O0)
        fin.readline()
        read_1darray(fin, value_segs_O0)
        fin.readline()
        read_1darray(fin, bases_O0)
        fin.readline()
        read_1darray(fin, popts_O0[:,0]) # a
        fin.readline()
        read_1darray(fin, popts_O0[:,1]) # b
        fin.readline()
        read_1darray(fin, popts_O0[:,2]) # c

    return \
        perc_segs_B0, value_segs_B0, bases_B0, popts_B0, \
        perc_segs_O0, value_segs_O0, bases_O0, popts_O0 


perc_segs_B['Global'], value_segs_B['Global'], bases_B['Global'], popts_B['Global'], \
perc_segs_O['Global'], value_segs_O['Global'], bases_O['Global'], popts_O['Global'] = \
    read_rescalingCoefficient('c37rescaling_global.dat')

perc_segs_B['Tropical'], value_segs_B['Tropical'], bases_B['Tropical'], popts_B['Tropical'], \
perc_segs_O['Tropical'], value_segs_O['Tropical'], bases_O['Tropical'], popts_O['Tropical'] = \
    read_rescalingCoefficient('c37rescaling_tropical.dat')

perc_segs_B['Extra-tropical'], value_segs_B['Extra-tropical'], bases_B['Extra-tropical'], popts_B['Extra-tropical'], \
perc_segs_O['Extra-tropical'], value_segs_O['Extra-tropical'], bases_O['Extra-tropical'], popts_O['Extra-tropical'] = \
    read_rescalingCoefficient('c37rescaling_extratropical.dat')


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
        a, b, c = popts[igap]
        base = bases[igap]
        percentiles = np.where((values>=Gap[0]) & (values<=Gap[1]), \
            inverse_exp(Gap[0], a, b, c, base), percentiles)

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

    print(perc_segs_B)
    print(obs_err_data)
    