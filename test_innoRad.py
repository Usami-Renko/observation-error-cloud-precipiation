'''
Description: test
Author: Hejun Xie
Date: 2022-04-07 19:12:11
LastEditors: Hejun Xie
LastEditTime: 2022-04-24 19:53:21
'''

import glob
import numpy as np
from innoRad import innoRad
import pickle

if __name__ == '__main__':

    innotbs_mwi_scatt  = []
    innotbs_mwi_direct = []

    # DA_DIRS = glob.glob('./201906/'+'[0-9]'*10)
    DA_DIRS = glob.glob('./202107/'+'[0-9]'*10)
    print(DA_DIRS)
    
    for DIR in DA_DIRS:
        innotbs_mwi_scatt.extend(  glob.glob('{}/innotbinno_fy3dmwi????????????.dat_scatt'.format(DIR))  )
        innotbs_mwi_direct.extend( glob.glob('{}/innotbinno_fy3dmwi????????????.dat_direct'.format(DIR)) )

    if False:
        innoRad_ds = innoRad(innotbs_mwi_scatt, innotbs_mwi_direct, 'mwri')
        with open('inno.pkl', 'wb') as fout:
            pickle.dump(innoRad_ds, fout)
    else:
        with open('inno.pkl', 'rb') as fin:
            innoRad_ds = pickle.load(fin)    
    
    '''
    1. QC
    '''
    innoRad_ds.qualityControl()
    # innoRad_ds.plotRecordQC(8, ['C37', 'BgQC', 'RFI'], 
    #     ['k','r','b'], ['202107190300', '202107220300'], 'QCrecord.png')
    # exit()
    innoRad_ds.remove_Blacked()
    
    '''
    2. Time slots select
    '''    
    # innoRad_ds.merge_data(startSlot='202107190300', endSlot='202107220300')
    # innoRad_ds.merge_data(startSlot='202107180300', endSlot='202107220300')
    innoRad_ds.merge_data()

    '''
    3.1 OMB distribution
    '''
    # for ich in range(0, 10):
    #     innoRad_ds.plot_OMB(ich, 'ch{}_OMB.png'.format(ich))

    '''
    3.2 O VS B regression
    '''
    # for ich in range(0, 10):
    #     innoRad_ds.plot_OVB(ich, 'ch{}_OVB.png'.format(ich))

    '''
    3.3 OMB space distribution
    '''
    # for ich in range(0, 10):
    #     innoRad_ds.plot_OMB_spacedist(ich, 'ch{}_spacedist.png'.format(ich),\
    #         dlatbin=4.0, dlonbin=4.0)
    
    # for ich in range(0, 10):
    #     innoRad_ds.plot_OMB_spacedist(ich, 'ch{}_spacedist_normed.png'.format(ich),\
    #         dlatbin=4.0, dlonbin=4.0, OMBkey='normed_OMB_BC_sct_c37_symme*')

    # for ich in range(0, 10):
    #     innoRad_ds.plot_OMB_scatter(ich, 'ch{}_OMB_scatter.png'.format(ich))

    # for ich in range(0, 10):
    #     innoRad_ds.plot_OMB_scatter(ich, 'ch{}_OMB_scatter_normed.png'.format(ich)),
    #     OMBkey='normed_OMB_BC_sct_c37_symme*')

    '''
    3.4 C37 space distribution
    '''
    # innoRad_ds.plot_c37_scatter(6, 'ch6_C37_scatter.png')
    # innoRad_ds.plot_c37_spacedist(6, 'ch6_C37_spacedist.png', dlatbin=4.0, dlonbin=4.0)
    
    '''
    3.5 observation error
    '''
    # c37bins = np.arange(-0.075, 0.725, 0.05)
    # for ich in range(0, 10):
    #     innoRad_ds.plot_c37_OMB(ich, 'ch{}_C37_OMB.png'.format(ich), c37bins)

    '''
    3.6 c37 percentile plot
    '''
    # innoRad_ds.plot_c37_percentile(8, 'ch8_C37_percentile.png')
    # innoRad_ds.plot_c37_pdf(8, 'ch8_c37_pdf.png')

    '''
    3.7 Nomalized OMB distribution
    '''
    # for ich in range(0, 10):
    #     innoRad_ds.plot_normalized_OMB(ich, 'ch{}_normed_OMB_pdf.png'.format(ich), OMBkey='OMB_BC_sct', 
    #         c37keys=['c37_symme*', 'c37_model*'], c37Colors=['r', 'b'], 
    #         c37Labels=[r'$\overline{C37}\ast$',r'$C37^{B}\ast$'])

    '''
    3.8 Plot BgQC
    '''
    # for ich in range(0, 10):
    #     innoRad_ds.plot_BgQC(ich, 'ch{}_BgQC.png'.format(ich), thresh_sigma=1.5)

