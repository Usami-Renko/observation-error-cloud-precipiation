'''
Description: class to Perform QC, observation error statistics
Author: Hejun Xie
Date: 2022-04-09 21:08:27
LastEditors: Hejun Xie
LastEditTime: 2022-05-13 20:41:50
'''
import logging
import os
import numpy as np
import copy
import scipy.stats as stats

from read_innotb import read_data
from utils import chkey, get_channel_str, get_slice, get_vminvmax, gaussian, cornerTag

np.warnings.filterwarnings('ignore') # to deal with NaNs in raw data before QC

def find_expsfit(perc_segs, percentiles, values):
    '''
    FIND MULTIPLE EXPONENTIAL FIT FOR DATA [X(PERCENTILES), Y(VALUES)]
    '''
    def expfit(x, a, b, c):
        base = (perc_range[0] + perc_range[1]) / 2.0
        return a * np.exp(b * (x - base)) + c

    from scipy.optimize import curve_fit
    perc_ranges = []
    perc_slices = []
    popts = []
    value_segs = []
    bases = []

    for perc_range in zip(perc_segs[:-1], perc_segs[1:]):
        perc_ranges.append(perc_range)
        perc_slices.append(get_slice(percentiles, perc_range))
    
    for perc_range, perc_slice in zip(perc_ranges, perc_slices):
        popt, pcov = curve_fit(expfit, percentiles[perc_slice], values[perc_slice], maxfev=5000)
        popts.append(popt)
        value_segs.append(expfit(percentiles[perc_slice.start], *popt))
        value_segs.append(expfit(percentiles[perc_slice.stop - 1], *popt))
        bases.append((perc_range[0] + perc_range[1]) / 2.0)

    return popts, perc_ranges, perc_slices, value_segs, bases


class innoRad(object):
    '''
        Description:
            A database class that contains innovations, QC BC informations.
            It can store results of cloud and precipitation simulations with RTTOV-SCATT
            and corresponding results of clear-sky simulations with RTTOV-DIRECT
            It can be initialized with two lists of innovation files taged as
            'scatt' (RTTOV-SCATT) and 'direct' (RTTOV), respectively, in a *zipped* manner
    '''
    
    def __init__(self, innofiles_scatt, innofiles_direct, instrument, latS=-40., latN=40.):
        '''
        Initialize it with  two lists of innovation files tagged as
        'scatt' (RTTOV-SCATT) and 'direct' (RTTOV), respectively, in a *zipped* manner
        '''
        
        '''
        0. Check and accept input parameters
        '''
        assert len(innofiles_scatt) == len(innofiles_direct), \
            "Please check each 'scatt' innovation files matches one 'direct' innovation file!"

        self.innofiles_scatt = innofiles_scatt
        self.innofiles_direct = innofiles_direct

        assert instrument in ['mwri', 'mwhs2'], \
            "Invalid instrument name '{}'!".format(instrument)

        self.instrument = instrument
        if instrument == 'mwri':
            self.num_chn = 10
        elif instrument == 'mwhs2':
            self.num_chn = 15

        '''
        1. Initialize data keys here
        '''
        self.mutualkeys = ['line', 'pos', 'lat', 'lon', 'surf-type', 'surf-height', 'BC', 'O', 'effecfrac']
        self.diffkeys = ['B', 'OMB_RAW', 'OMB_BC'] # variables that have different values for RTTOV-SCATT and RTTOV-DIRECT
        self.datakeys = self.mutualkeys + \
            [(key+'_sct') for key in self.diffkeys] + [(key+'_drt') for key in self.diffkeys]

        self.c37 = False # Flag indicating if c37 has been activated
        self.c37keys = ['c37_obser', 'c37_model', 'c37_symme', 'c37_model*', 'c37_symme*'] # will be added to datakeys if self.c37 set to True

        self.OMBkeys = ['OMB_RAW_sct', 'OMB_RAW_drt', 'OMB_BC_sct', 'OMB_BC_drt'] # declare OMBkeys here
        
        self.categkeys = [] # store datakey where varaiblaA is categorized by variableB, will not be added to datakeys  
        self.normkeys = [] # store datakey where variableA is normalized by variableB, will be added to datakeys

        '''
        2. Set plot box (& default black latitude settings) 
        '''
        self.latS = latS
        self.latN = latN

        '''
        3.1 Variables populated in self.read_innodata()   
        '''
        self.windowSlots = []
        self.innodata = dict()
        self.nrad = dict()

        '''
        3.2 Variables populated in self.qualityControl()
        '''
        self.qcflag = dict()
        self.qcRecords = [] # A list of blacked record (numpy.ndArray of size self.nrad)
        self.qcChannels = [] # A list of blacked channel (An integer in range [0, self.num_chn-1] or 'all') 
        self.qcLabels = [] # A list of QC label (string)
        self.qcWindowSlots = [] # A list of window Slot (string)
        self.valid_qcLbels = [] # ['C37', 'Land-Coast', 'High-latitude', 'Land-sea-Contamination', 'RFI', 'Abnormal']
        self.Tropical_BOXES = [] # for bi-mode rescaling coefficients
        
        '''
        3.3 Variables populated in self.merge_data()
        '''
        self.merged_innodata = dict()
        self.startSlot = None
        self.endSlot = None
        
        '''
        4. Read in data here
        '''
        self.read_innodata()
        
        '''
        5. logging initialized here
        '''
        fmt="%(asctime)s [line:%(lineno)d] - %(levelname)s: %(message)s"
        self.logfile = '{}:{}-{}.log'.format(instrument, self.windowSlots[0], self.windowSlots[-1])
        logging.basicConfig(filename=self.logfile, filemode='w', level=logging.INFO, format=fmt)
    
    def read_innodata(self):
        '''
        Read innovation files
        Fill self.innodata & self.nrad
        '''

        for innofile_scatt, innofile_direct in zip(self.innofiles_scatt, self.innofiles_direct):
            datetime_windowSlot = os.path.basename(innofile_scatt).split('.')[-2][-12:]
            self.windowSlots.append(datetime_windowSlot)
            
            tempdata_scatt = read_data(innofile_scatt, self.num_chn)
            tempdata_direct = read_data(innofile_direct, self.num_chn)

            self.nrad[datetime_windowSlot] = len(tempdata_scatt[chkey(0)]['line'])

            tempdata_windowSlot = dict()
            for ich in range(self.num_chn):
                tempdata_windowSlot[chkey(ich)] = dict()
            
            for key in self.mutualkeys:
                for ich in range(self.num_chn):
                    tempdata_windowSlot[chkey(ich)][key] = tempdata_scatt[chkey(ich)][key]
            
            for key in self.diffkeys:
                for ich in range(self.num_chn):
                    tempdata_windowSlot[chkey(ich)][key+'_sct'] = tempdata_scatt[chkey(ich)][key]
                    tempdata_windowSlot[chkey(ich)][key+'_drt'] = tempdata_direct[chkey(ich)][key]
            
            self.innodata[datetime_windowSlot] = tempdata_windowSlot
    
    def init_qcflag(self):
        '''
        Initialize self.qcflag
        '''
        for windowSlot in self.innodata.keys():
            '''
            QC flag for total Pixel (PixQC)
            '''
            self.qcflag[windowSlot] = dict()
            self.qcflag[windowSlot]['PixQC'] = np.full((self.nrad[windowSlot]), False, dtype=bool) 
            
            '''
            QC flag for that channel (BgQC)
            '''
            for ich in range(self.num_chn):
                self.qcflag[windowSlot][chkey(ich)] = np.full((self.nrad[windowSlot]), False, dtype=bool)
    
    def get_data(self, windowSlot, ch_no, datakey):
        '''
        Get data of given a given datakey
        !!! To avoid modifying the original data, please use this method to get a copy of data
        '''
        assert datakey in self.datakeys, \
            "Invalid datakey '{}'!".format(datakey)

        assert windowSlot in self.windowSlots, \
            "Invalid windowSlot '{}'!".format(windowSlot)
        
        return copy.copy(self.innodata[windowSlot][chkey(ch_no)][datakey]) # protect original data
    
    def set_data(self, windowSlot, ch_no, datakey, data):
        '''
        Get data of given a given datakey
        !!! To avoid modifying the setted data after setting, use this method
        '''
        assert datakey in self.datakeys, \
            "Invalid datakey '{}'!".format(datakey)

        assert windowSlot in self.windowSlots, \
            "Invalid windowSlot '{}'!".format(windowSlot)
        
        self.innodata[windowSlot][chkey(ch_no)][datakey] = copy.copy(data) # protect setted data

    def get_merged_data(self, ch_no, datakey):
        '''
        Get merged data of given a given datakey
        !!! To avoid modifying the original data, please use this method to get a copy of data
        '''
        assert (datakey in self.datakeys) or (datakey in self.categkeys), \
            "Invalid datakey '{}'!".format(datakey)

        return copy.copy(self.merged_innodata[chkey(ch_no)][datakey]) # protect original data
    
    def set_merged_data(self, ch_no, datakey, data):
        '''
        Set merged data of given a given datakey
        !!! To avoid modifying the setted data after setting, use this method
        '''
        assert (datakey in self.datakeys) or (datakey in self.categkeys), \
            "Invalid datakey '{}'!".format(datakey)

        self.merged_innodata[chkey(ch_no)][datakey] = copy.copy(data) # protect setted data

    def calc_c37(self, rescale='mono', lqcRecord=False, qcLabel='C37'):
        '''
        For MWRI only !!!
        Calculates the cloud fraction from 37.0GHz V pol. - H pol.
        rescale = 'mono': global rescaling
        rescale = 'bi': tropical and extratropical rescaling

        '''
        self.c37 = True
        self.datakeys = self.datakeys + self.c37keys

        for windowSlot in self.innodata.keys():
        
            T_V_O_RAW = self.get_data(windowSlot, 6, 'O')
            T_H_O_RAW = self.get_data(windowSlot, 7, 'O')

            V_BC = self.get_data(windowSlot, 6, 'BC')
            H_BC = self.get_data(windowSlot, 7, 'BC')
            
            T_V_Bsct = self.get_data(windowSlot, 6, 'B_sct')
            T_H_Bsct = self.get_data(windowSlot, 7, 'B_sct')

            T_V_Bdrt = self.get_data(windowSlot, 6, 'B_drt')
            T_H_Bdrt = self.get_data(windowSlot, 7, 'B_drt')

            '''
            Perform Bias correction here
            '''
            T_V_O = T_V_O_RAW - V_BC
            T_H_O = T_H_O_RAW - H_BC

            '''
            Perform QC for c37 calc here
            '''
            c37Flag = ((T_V_Bdrt - T_H_Bdrt) < 0.1) | \
                      np.isnan(T_V_O) | np.isnan(T_H_O) |\
                      np.isnan(T_V_Bsct) | np.isnan(T_H_Bsct) |\
                      np.isnan(T_V_Bdrt) | np.isnan(T_H_Bdrt)
            
            self.recordQC(lqcRecord, c37Flag, 'all', qcLabel, windowSlot)
            self.qcflag[windowSlot]['PixQC'] |= c37Flag
            if np.sum(c37Flag):
                logging.info('DT37 QC blacklists {:>4}/{:>4} Pixels at {}'.format(np.sum(c37Flag), np.size(c37Flag), windowSlot))

            
            '''
            Calculate C37
            '''
            P37_model = (T_V_Bsct - T_H_Bsct) / (T_V_Bdrt - T_H_Bdrt)
            P37_obser = (T_V_O - T_H_O) / (T_V_Bdrt - T_H_Bdrt)
            C37_model = 1.0 - P37_model
            C37_obser = 1.0 - P37_obser 

            '''
            Clip data to [-0.1, 1.0]
            '''
            C37_model = np.clip(C37_model, -0.1, 1.0)
            C37_obser = np.clip(C37_obser, -0.1, 1.0)

            from obs_err_new import rescale_B2O
            if rescale == 'mono':
                C37_model_rescaled = rescale_B2O(C37_model, region='Global')
            elif rescale == 'bi':
                C37_model_rescaled_T = rescale_B2O(C37_model, region='Tropical')
                C37_model_rescaled_ET = rescale_B2O(C37_model, region='Extra-tropical')
                '''
                Get Tropical Flag: inT
                '''
                lat = self.get_data(windowSlot, 0, 'lat')
                lon = self.get_data(windowSlot, 0, 'lon')
                inT = np.full(lat.shape, False, dtype=bool)
                for box in self.Tropical_BOXES:
                    latS, latN, lonW, lonE = box
                    inT |= ((lat > latS) & (lat < latN) & \
                            (lon > lonW) & (lon < lonE))
                C37_model_rescaled = np.where(inT, C37_model_rescaled_T, C37_model_rescaled_ET) 
            else:
                exit('Unknown rescale mode: {}, exit'.format(rescale))

            for ich in range(self.num_chn):
                self.set_data(windowSlot, ich, 'c37_model', C37_model)
                self.set_data(windowSlot, ich, 'c37_model*', C37_model_rescaled) 
                self.set_data(windowSlot, ich, 'c37_obser', C37_obser)
                self.set_data(windowSlot, ich, 'c37_symme', (C37_model + C37_obser) / 2.0)
                self.set_data(windowSlot, ich, 'c37_symme*', (C37_model_rescaled + C37_obser) / 2.0)

    def calc_normalized_OMB(self, OMBkey='OMB_BC_sct', c37key='c37_symme*'):
        '''
        calculate OMB normalized by observation error
        set datakey as normed_<OMBkey>_<c37key>
        '''
        assert OMBkey in self.OMBkeys, \
            "Invalid OMBkey '{}'!".format(OMBkey)
        
        assert c37key in self.c37keys, \
            "Invalid OMBkey '{}'!".format(c37key)
        
        normkey = self.get_normed_key(OMBkey, c37key)
        self.normkeys.append(normkey)
        self.datakeys.append(normkey)

        from obs_err import calc_obs_err

        for windowSlot in self.windowSlots:
            c37 = self.get_data(windowSlot, 0, c37key)
            for ich in range(self.num_chn):
                OMB = self.get_data(windowSlot, ich, OMBkey)
                obserr = calc_obs_err(ich, c37)
                self.set_data(windowSlot, ich, normkey, OMB / obserr)
    
    def qualityControl(self):
        '''
        Perform QC (Quality Control)
        '''
        self.init_qcflag()

        # 0. Black region
        from obs_err import Tropical_BOXES
        self.Tropical_BOXES = Tropical_BOXES
        # self.black_region(self.Tropical_BOXES, inverse=True)
        
        # 1. PixQC
        self.black_landCoast()
        self.black_lat(latS=self.latS, latN=self.latN)
        if self.instrument == 'mwri':
            # self.black_landSeaContamination(lqcRecord=True)
            self.calc_c37(rescale='bi', lqcRecord=True)
        
        # 2. Channel QC
        self.black_abnormal(lower_bound=70., upper_bound=320.)
        self.calc_normalized_OMB(OMBkey='OMB_BC_sct', c37key='c37_symme*')
        self.calc_normalized_OMB(OMBkey='OMB_BC_sct', c37key='c37_model*')
        # self.black_BgQC(thresh_sigma=2.5, OMBkey='OMB_BC_sct', c37key='c37_symme*', lqcRecord=True)
        if self.instrument == 'mwri':
            self.black_RadioFreqInterf(lqcRecord=True)

        # 3. Apply PixQC to Channel QC
        self.apply_PixQC()
    
    def recordQC(self, lqcRecord, qcRecord, qcChannel, qcLabel, qcWindowSlot):
        '''
        1. add qc information to self.qcflag
        '''
        if qcChannel == 'all':
            self.qcflag[qcWindowSlot]['PixQC'] |= qcRecord
            logging.info('{} blacklists {:>4}/{:>4} Pixels at {}'.format(qcLabel, np.sum(qcRecord), np.size(qcRecord), qcWindowSlot))
        elif qcChannel in range(self.num_chn):
            self.qcflag[qcWindowSlot][chkey(qcChannel)] |= qcRecord
            logging.info('{} blacklists {:>4}/{:>4} Pixels at {}/{}'.format(qcLabel, np.sum(qcRecord), np.size(qcRecord), qcWindowSlot, chkey(qcChannel)))
        else:
            raise Exception("Invalid qcChannel {}".format(qcChannel))

        '''
        2. Record the QC information for plotting if lqcRecord
        '''
        if not lqcRecord:
            return
        if qcLabel not in self.valid_qcLbels:
            self.valid_qcLbels.append(qcLabel)
        
        self.qcRecords.append(qcRecord)
        self.qcChannels.append(qcChannel)
        self.qcLabels.append(qcLabel)
        self.qcWindowSlots.append(qcWindowSlot)
    
    def plotRecordQC(self, qcChannel, qcLabels, qcColors, qcwindowSlotRange, outpic):
        '''
        Plot Recorded QC scatter of several QC types indicated by qcLabels
        Scatters are plotted in qcColors
        Plotted scatters are assumed to be in qcwindowSlotRange and be assigned qcChannel
        '''

        for qcLabel in qcLabels:
            assert qcLabel in self.valid_qcLbels, \
                "Invalid qcLabel {}!".format(qcLabel)

        import matplotlib.pyplot as plt
        from collections import OrderedDict

        figsize = (14,6.0)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        map = self.get_latband_map(ax)

        for Record, Channel, Label, WindowSlot in zip(self.qcRecords, self.qcChannels, self.qcLabels, self.qcWindowSlots):
            if  (Channel not in ['all', qcChannel]) or \
                (Label not in qcLabels) or \
                (WindowSlot < qcwindowSlotRange[0] or WindowSlot > qcwindowSlotRange[1]) or \
                (np.sum(Record) == 0):
                continue
            

            Label_idx = qcLabels.index(Label)
            color = qcColors[Label_idx]

            latTag = self.get_data(WindowSlot, qcChannel, 'lat')
            lonTag = self.get_data(WindowSlot, qcChannel, 'lon')

            # do filter
            latTag = latTag[Record]
            lonTag = lonTag[Record]

            x, y = map(lonTag, latTag)
            map.scatter(x, y, marker='D', s=3, c=color, label=Label)

        '''
        1. remove black information on lands
        '''
        map.fillcontinents(color='white')
        
        '''
        To reduce redundant labels
        '''
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=14, markerscale=3.0, loc='upper left')

        cornerText = get_channel_str(self.instrument, qcChannel) + ' [' + qcwindowSlotRange[0] + '-' + qcwindowSlotRange[1] + ']'
        cornerTag(ax, cornerText, 25)

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()

    def black_BgQC(self, thresh_sigma=2.5, OMBkey='OMB_BC_sct', c37key='c37_symme*', lqcRecord=False, qcLabel='BgQC'):
        '''
        Do BgQC [Background QC] here
        '''
        normkey = self.get_normed_key(OMBkey, c37key)
        assert normkey in self.normkeys, \
            "Invalid normkey '{}'!".format(normkey)
        
        for windowSlot in self.windowSlots:
            for ich in range(self.num_chn):
                normedOMB = self.get_data(windowSlot, ich, normkey)
                BgQC = (normedOMB > thresh_sigma) | (normedOMB < -thresh_sigma)
                self.recordQC(lqcRecord, BgQC, ich, qcLabel, windowSlot)

    def black_landCoast(self, lqcRecord=False, qcLabel='Land-Coast'):
        '''
        Black continent radiances
        check surf-type of each pixel
            surf-type=0: ocean
            surf-type=1: coast
            surf-type=2: land
        '''
        for windowSlot in self.innodata.keys():
            surf_type = self.get_data(windowSlot, 0, 'surf-type')
            landCoastQC = (surf_type != 0)
            self.recordQC(lqcRecord, landCoastQC, 'all', qcLabel, windowSlot)

    def black_lat(self, lqcRecord=False, qcLabel='High-latitude', latS=None, latN=None):
        '''
        Black high latitude radiances
        '''
        if latS is None:
            latS = self.latS
        else:
            self.latS = latS
        
        if latN is None:
            latN = self.latN
        else:
            self.latN = latN
        
        for windowSlot in self.innodata.keys():
            lat = self.get_data(windowSlot, 0, 'lat')
            latQC = (lat < latS) | (lat > latN)
            self.recordQC(lqcRecord, latQC, 'all', qcLabel, windowSlot)
    
    def black_landSeaContamination(self, lqcRecord=False, qcLabel='Land-sea-Contamination'):
        '''
        For MWRI only !!!
        Black land/sea contaminated pixels Using information from 10.65GHz H/V
        '''
        for windowSlot in self.innodata.keys():
            ch10_65V = self.get_data(windowSlot, 0, 'O')
            ch10_65H = self.get_data(windowSlot, 1, 'O')
            landSeaContaQC = (ch10_65V > 175.) | (ch10_65H > 95.)
            self.recordQC(lqcRecord, landSeaContaQC, 'all', qcLabel, windowSlot)
    
    def black_RadioFreqInterf(self, lqcRecord=False, qcLabel='RFI'):
        '''
        For MWRI only !!!
        Black Radio Frequency Interference [PRF] channels at 18.7GHz H/V
        '''
        for windowSlot in self.innodata.keys():
            ch18_7V = self.get_data(windowSlot, 2, 'O') 
            ch18_7H = self.get_data(windowSlot, 3, 'O') 
            ch23_8V = self.get_data(windowSlot, 4, 'O') 
            ch23_8H = self.get_data(windowSlot, 5, 'O') 
            vRFI_QC = (ch18_7V > ch23_8V)
            hRFI_QC = (ch18_7H > ch23_8H)

            self.recordQC(lqcRecord, vRFI_QC, 2, qcLabel, windowSlot)
            self.recordQC(lqcRecord, hRFI_QC, 3, qcLabel, windowSlot)

    def black_abnormal(self, lqcRecord=False, qcLabel='Abnormal', upper_bound=320., lower_bound=70.):
        '''
        Black abnormal observation or simulations
        '''
        for windowSlot in self.innodata.keys():
            for ich in range(self.num_chn):
                B_drt = self.get_data(windowSlot, ich, 'B_drt') 
                B_sct = self.get_data(windowSlot, ich, 'B_sct') 
                O = self.get_data(windowSlot, ich, 'O')
                AbnQC = np.isnan(B_drt) | np.isnan(B_drt) | np.isnan(O) |\
                        (B_drt > upper_bound) | (B_drt < lower_bound) | (B_sct > upper_bound) | (B_sct < lower_bound) |\
                        (O > upper_bound) | (O < lower_bound) 
                
                self.recordQC(lqcRecord, AbnQC, ich, qcLabel, windowSlot)
    
    def black_region(self, boxes, inverse=False, lqcRecord=False, qcLabel='Region'):
        '''
        Black selected regions
        boxes: a list of box for selected region
        box: [latS, latN, lonW, lonE]
        '''
        for windowSlot in self.innodata.keys():
            lat = self.get_data(windowSlot, 0, 'lat')
            lon = self.get_data(windowSlot, 0, 'lon')

            regionQC = np.full(lat.shape, False, dtype=bool)
            for box in boxes: 
                boxQC = (lat > box[0]) & (lat < box[1]) & \
                        (lon > box[2]) & (lon < box[3])
                regionQC |= boxQC # black regions inside boxes
            
            if inverse: # black regions outside boxes
                regionQC = ~regionQC
            
            self.recordQC(lqcRecord, regionQC, 'all', qcLabel, windowSlot)

    def apply_PixQC(self):
        '''
        Apply Pixel QC to channels
        '''
        for windowSlot in self.innodata.keys():
            for ich in range(self.num_chn):
                self.qcflag[windowSlot][chkey(ich)] |= self.qcflag[windowSlot]['PixQC']
    
    def remove_Blacked(self):
        '''
        Remove those blacked channels
        '''
        for windowSlot in self.innodata.keys():
            for ich in range(self.num_chn):
                qc_flag = self.qcflag[windowSlot][chkey(ich)]
                for datakey in self.datakeys:                
                    self.innodata[windowSlot][chkey(ich)][datakey] = \
                        self.innodata[windowSlot][chkey(ich)][datakey][~qc_flag]

    def merge_data(self, startSlot=None, endSlot=None):
        '''
        merge slots and make merged data
        '''
        if startSlot is not None:
            start_index = self.windowSlots.index(startSlot)
        else:
            start_index = None
            startSlot = self.windowSlots[0]
        if endSlot is not None:
            end_index   = self.windowSlots.index(endSlot)
        else:
            end_index = None
            endSlot = self.windowSlots[-1]

        # 1.initialize
        for ich in range(self.num_chn):
            self.merged_innodata[chkey(ich)] = dict()
            for datakey in self.datakeys:
                self.merged_innodata[chkey(ich)][datakey] = []
        
        # 2.make list
        for windowSlot in self.windowSlots[start_index:end_index]:
            for ich in range(self.num_chn):
                for datakey in self.datakeys:
                    self.merged_innodata[chkey(ich)][datakey].append(
                        self.innodata[windowSlot][chkey(ich)][datakey]
                    )

        # 3.merge with numpy.hstack
        for ich in range(self.num_chn):
            for datakey in self.datakeys:
                self.merged_innodata[chkey(ich)][datakey] = \
                    np.hstack(self.merged_innodata[chkey(ich)][datakey])
        
        self.startSlot = startSlot
        self.endSlot = endSlot

    def calc_hist(self, ch_no, datakey, bins, norm=1.0):
        '''
        Get histogram distribution of a given datakey
        of self.merged_innodata
        '''
        data = self.get_merged_data(ch_no, datakey) / norm
        bin_center = (bins[1:] + bins[:-1]) / 2.0
        return np.histogram(data, bins)[0], bin_center       
    
    def calc_meanStdRmse(self, ch_no, datakey):
        '''
        Get Mean, Standard Deviation, Root Mean Square Error (RMSE) of a given datakey 
        of self.merged_innodata
        '''
        data = self.get_merged_data(ch_no, datakey)
        return np.mean(data), np.std(data), np.sqrt(np.mean(data*data))
    
    def calc_percentile(self, ch_no, datakey, percentiles):
        '''
        Get percentiles of a givens datakey
        '''
        data = self.get_merged_data(ch_no, datakey)
        return np.percentile(data, percentiles)
    
    def pool_data(self, data, latTag, lonTag, dlatbin, dlonbin, reduceMethod='mean', returnCenter=True):
        '''
        Pool data in dlatbin * dlonbin squares
        and return the pooled array
        '''
        latbins = np.arange(self.latS, self.latN + dlatbin, dlatbin)
        lonbins = np.arange(0.,  360. + dlonbin, dlonbin)

        latCenter = (latbins[1:] + latbins[:-1]) / 2.0
        lonCenter = (lonbins[1:] + lonbins[:-1]) / 2.0

        pooledData = np.zeros((len(latbins)-1, len(lonbins)-1), dtype='float32')

        for ilat in range(len(latbins)-1):
            latRange = (latbins[ilat], latbins[ilat+1])
            for ilon in range(len(lonbins)-1):
                lonRange = (lonbins[ilon], lonbins[ilon+1])
                squareFlag = (latTag >= latRange[0]) & (latTag < latRange[1]) & \
                             (lonTag >= lonRange[0]) & (lonTag < lonRange[1])
                dataInSquare = data[squareFlag]
                if reduceMethod == 'mean':
                    pooledData[ilat, ilon] = np.mean(dataInSquare)
                if reduceMethod == 'Median':
                    pooledData[ilat, ilon] = np.median(dataInSquare)
        
        if returnCenter:
            return pooledData, latCenter, lonCenter
        else:
            return pooledData, latbins, lonbins   
        
    def get_categ_key(self, categed_datakey, categing_datakey, categ_range):
        '''
        Get categorized datakey
        '''
        return '{}_{}_{:>+.2f}_{:>+.2f}'.format(categed_datakey, categing_datakey, categ_range[0], categ_range[1])
    
    def get_normed_key(self, OMBkey, c37key):
        '''
        Get normalized c37 datakey
        '''
        return 'normed_{}_{}'.format(OMBkey, c37key)

    def categ_with_c37(self, ch_no, c37key, datakey, c37bins):
        '''
        Get categorized data named by datakey using c37
        '''
        assert c37key in self.c37keys, \
            "Invalid c37key '{}'!".format(c37key)
            
        data = self.get_merged_data(ch_no, datakey)
        c37Tag = self.get_merged_data(ch_no, c37key)
        volumn = np.zeros((len(c37bins)-1), dtype='float32')

        for ic37bin in range(len(c37bins)-1):
            c37Range = (c37bins[ic37bin], c37bins[ic37bin+1])
            RangeFlag = (c37Tag > c37Range[0]) & (c37Tag < c37Range[1])
            volumn[ic37bin] = np.sum(RangeFlag)
            dataInc37Range = data[RangeFlag]
            c37RangeDatakey = self.get_categ_key(datakey, c37key, c37Range)            
            self.categkeys.append(c37RangeDatakey)
            self.set_merged_data(ch_no, c37RangeDatakey, dataInc37Range)

        return volumn

    def plot_c37_OMB(self, ch_no, outpic, c37bins):
        '''
        Plot the bias and Rms of categorized OMB using 3 definitions
        of C37  
        '''
        datakey = 'OMB_BC_sct'

        B_datakey = 'c37_model*'
        S_datakey = 'c37_symme*'

        volumn_c37_obser = self.categ_with_c37(ch_no, 'c37_obser', datakey, c37bins)
        volumn_c37_model = self.categ_with_c37(ch_no, B_datakey, datakey, c37bins)
        volumn_c37_symme = self.categ_with_c37(ch_no, S_datakey, datakey, c37bins)

        # get percentile and labels
        covers = [90., 70., 50., 30., 10.]
        percentiles = [(100.-cover)/2.0 for cover in covers] + [50.] + [100.-(100.-cover)/2.0 for cover in covers[::-1]]
        cover_labels = ["{}\%".format(cover) for cover in covers]


        bias_c37obser = np.zeros((len(c37bins)-1), dtype='float32')
        bias_c37model = np.zeros((len(c37bins)-1), dtype='float32')
        bias_c37symme = np.zeros((len(c37bins)-1), dtype='float32')

        std_c37obser = np.zeros((len(c37bins)-1), dtype='float32')
        std_c37model = np.zeros((len(c37bins)-1), dtype='float32')
        std_c37symme = np.zeros((len(c37bins)-1), dtype='float32')

        perc_c37symme = np.zeros(((len(c37bins)+1), len(percentiles)), dtype='float32')
        perc_c37bins = np.zeros(((len(c37bins)+1)), dtype='float32')

        for ic37bin in range(len(c37bins)-1):
            c37Range = (c37bins[ic37bin], c37bins[ic37bin+1])
            c37RangeDatakey = self.get_categ_key(datakey, 'c37_obser', c37Range)
            bias_c37obser[ic37bin], std_c37obser[ic37bin], _  = self.calc_meanStdRmse(ch_no, c37RangeDatakey)
            
            c37RangeDatakey = self.get_categ_key(datakey, B_datakey, c37Range) 
            bias_c37model[ic37bin], std_c37model[ic37bin], _  = self.calc_meanStdRmse(ch_no, c37RangeDatakey)

            c37RangeDatakey = self.get_categ_key(datakey, S_datakey, c37Range) 
            bias_c37symme[ic37bin], std_c37symme[ic37bin], _  = self.calc_meanStdRmse(ch_no, c37RangeDatakey)
            perc_c37symme[ic37bin+1,:] = self.calc_percentile(ch_no, c37RangeDatakey, percentiles)

        c37binCenter = (c37bins[1:] + c37bins[:-1]) / 2.0

        perc_c37symme[0,:] = perc_c37symme[1,:]
        perc_c37symme[-1:,:] = perc_c37symme[-2,:]
        perc_c37bins[1:-1] = c37binCenter
        perc_c37bins[0] = c37bins[0]
        perc_c37bins[-1] = c37bins[-1] 

        # print(perc_c37bins)
        # exit()
        
        xlim = (c37bins[0], c37bins[-1])

        '''
        Start plot bias and STD
        '''
        figsize = (6.0,5.0)
        mew=3.0
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize[0]*2.0,figsize[1]*2.0), sharex=True)


        '''
        1. Bias
        '''
        axes[0][0].plot(c37binCenter, bias_c37obser, marker='x', mew=mew, label=r'$C_{37}^{O}$', color='r')
        axes[0][0].plot(c37binCenter, bias_c37model, marker='x', mew=mew, label=r'$C_{37}^{S}$', color='b')
        axes[0][0].plot(c37binCenter, bias_c37symme, marker='x', mew=mew, label=r'$\overline{C_{37}}$', color='k')
        cornerText = get_channel_str(self.instrument, ch_no)
        cornerTag(axes[0][0], cornerText, 20)
        axes[0][0].set_xlim(xlim[0], xlim[1])
        ylim_tmp = axes[0][0].get_ylim()
        ylim_abs = max(np.abs(ylim_tmp[0]), np.abs(ylim_tmp[1]))
        axes[0][0].set_ylim(-ylim_abs, ylim_abs)
        axes[0][0].hlines(y=0.0, xmin=c37bins[0], xmax=c37bins[-1], colors='r', ls='--')
        axes[0][0].set_title(r'$\textbf{a) Bias [K]}$', fontsize=18)
        axes[0][0].legend(loc='upper left', markerscale=1.5)

        '''
        2. STD
        '''
        axes[0][1].plot(c37binCenter, std_c37obser, marker='x', mew=mew, label=r'$C_{37}^{O}$', color='r')
        axes[0][1].plot(c37binCenter, std_c37model, marker='x', mew=mew, label=r'$C_{37}^{S}$', color='b')
        axes[0][1].plot(c37binCenter, std_c37symme, marker='x', mew=mew, label=r'$\overline{C_{37}}$', color='k')
        
        axes[0][1].set_xlim(xlim[0], xlim[1])
        axes[0][1].set_ylim(0.0, max(np.max(std_c37obser), np.max(std_c37model), np.max(std_c37symme))*1.3)
        axes[0][1].set_title(r'$\textbf{b) Standard Deviation [K]}$', fontsize=18)

        '''
        Perform Linear regression here and get observation error model for cloud and precipitation regions
        Reference:
        [1]. Geer, Alan J., and Peter Bauer. "Observation errors in all-sky data assimilation." 
        Quarterly Journal of the Royal Meteorological Society 137.661 (2011): 2024-2037.

        Here We assume: 
            1) alpha=1.0 
               (that is all error is attibuted to observation error, not from background error, see [1].)
            2) Cclr = 0.0;
               Ccld = c37binCenter[np.argmax(std_c37obser)]
            3) g2(c37) =  | gclr (c37<Cclr)
                          | k * c37 + gclr (Cclr<=c37<=Ccld)
                          | gcld (c37>Ccld)
        Here we get observation error structure:
            Ccld, gclr, gcld, k 
        '''
        Cclr = 0.0
        Ccld = c37binCenter[np.argmax(std_c37symme)]
        reg_slice = get_slice(c37binCenter, [Cclr, Ccld])
        # result = stats.linregress(c37binCenter[reg_slice], y=std_c37symme[reg_slice])
        # slope, intercept, rvalue, pvalue, stderr = result
        intercept = std_c37symme[reg_slice.start] # at Cclr
        slope = np.max( (std_c37symme[reg_slice] - intercept) / (c37binCenter[reg_slice] - Cclr) ) 
        Ccld = Cclr + (std_c37symme[reg_slice.stop-1] - intercept) / slope # update Ccld
        
        k = slope
        gclr = intercept
        gcld = intercept + slope * Ccld

        axes[0][1].hlines(y=gcld, xmin=Ccld, xmax=c37bins[-1], colors='k', ls='-.', lw=1.5)
        axes[0][1].hlines(y=gclr, xmin=c37bins[0], xmax=Cclr,  colors='k', ls='-.', lw=1.5)
        axes[0][1].plot([Cclr, Ccld], [Cclr*slope + intercept, Ccld*slope + intercept], color='k', ls='-.', lw=1.5)

        ylim = axes[0][1].get_ylim()
        axes[0][1].vlines(x=Cclr, ymin=ylim[0], ymax=ylim[1],  colors='k', ls='--', lw=2.0)
        axes[0][1].vlines(x=Ccld, ymin=ylim[0], ymax=ylim[1],  colors='k', ls='--', lw=2.0)

        axes[0][1].text(Cclr, -0.5, r'$C_{clr}$', ha='center', va='top', size=20)
        axes[0][1].text(Ccld, -0.5, r'$C_{cld}$', ha='center', va='top', size=20)

        # axes[0][1].text(c37bins[0] + 0.015, gclr + 3.0, r'$g_{clr}$', ha='left', va='bottom', size=20)
        # axes[0][1].text(c37bins[-1]- 0.03, gcld + 0.5, r'$g_{cld}$', ha='right', va='bottom', size=20)

        print('ch:{} Ccld={:>.3f}, gclr={:>.2f}, gcld={:>.2f}, k={:>.2f}'.format(ch_no, Ccld, gclr, gcld, k))

        '''
        3. Percentile
        '''
        from matplotlib import cm
        colormap = cm.rainbow
        axes[1][0].set_xlabel(r'$C_{37}$')
        for icover, cover_label in enumerate(cover_labels):
            axes[1][0].fill_between(perc_c37bins, perc_c37symme[:,icover], perc_c37symme[:,-icover-1], \
                facecolor=colormap((icover+1)/len(covers)), label=cover_label)
        axes[1][0].plot(c37binCenter, perc_c37symme[1:-1,len(percentiles)//2], color='k', label='Median', mew=3.0, marker='x')

        ylim_tmp = axes[1][0].get_ylim()
        ylim_abs = max(np.abs(ylim_tmp[0]), np.abs(ylim_tmp[1]))
        axes[1][0].set_ylim(-ylim_abs, ylim_abs)

        axes[1][0].text(0.05, 0.05, r'FG Departure Over $\overline{C_{37}}$', ha='left', va='bottom', size=20, transform=axes[1][0].transAxes)

        axes[1][0].legend(loc='upper left', ncol=3, fontsize=15)
        axes[1][0].set_title(r'$\textbf{c) First Guess Departure [K]}$', fontsize=18)
        axes[1][0].hlines(y=0.0, xmin=c37bins[0], xmax=c37bins[-1], colors='r', ls='--')

        '''
        4. Volumn
        '''
        barwidth=0.01
        axes[1][1].set_xlabel(r'$C_{37}$')
        axes[1][1].bar(c37binCenter-barwidth, volumn_c37_obser, barwidth, label=r'$C_{37}^{O}$', color='r')
        axes[1][1].bar(c37binCenter+barwidth, volumn_c37_model, barwidth, label=r'$C_{37}^{S}$', color='b')
        axes[1][1].bar(c37binCenter, volumn_c37_symme, barwidth, label=r'$\overline{C_{37}}$', color='k')
        axes[1][1].set_title(r'$\textbf{d) Numbers Of Observations [-]}$', fontsize=18)
        axes[1][1].set_yscale('log')
        axes[1][1].set_ylim(1e+1, 1e+6)
        axes[1][1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(outpic)
        plt.close()
    
    def plot_c37_percentile(self, ch_no, outpic):
        '''
        Plot c37 percentile
        '''
        percentiles = np.linspace(0., 100., 10001)
        
        '''
        Cloud Tail inset plot settings here:
            [xmin, xmax, ymin, ymax]
        '''
        cloud_tail = [95., 100., 0.3, 1.0]
        
        '''
        Get C37 data percentile data
        '''
        C37_O_perc = self.calc_percentile(ch_no, 'c37_obser', percentiles)
        C37_B_perc = self.calc_percentile(ch_no, 'c37_model', percentiles)

        '''
        Get fit parameters here (online)
        To modify region, uncomment self.black_region() in self.qualityControl() 
        '''
        online_coeff_region = 'Global'
        online_coeff_region = 'Tropical'
        perc_segs_B = [0., .5, 5., 60., 80., 90., 95., 98., 99.5, 99.9, 100.]
        popts_B, perc_ranges_B, perc_slices_B, value_segs_B, bases_B = find_expsfit(perc_segs_B, percentiles, C37_B_perc)
        # perc_segs_O = [0., 2.5, 5., 10., 20., 40., 70., 90., 95., 100.]
        perc_segs_O = [0., 2.5, 5., 10., 20., 40., 70., 85., 92., 96., 99., 100.]
        popts_O, perc_ranges_O, perc_slices_O, value_segs_O, bases_O = find_expsfit(perc_segs_O, percentiles, C37_O_perc)
        # np.set_printoptions(formatter={'float': '{:+.6e}'.format}) # output coefficient in terminal
        # print(np.array(perc_segs_O))
        # print(np.array(value_segs_O))
        # print(np.array(bases_O))
        # print(popts_O)
        # exit()

        # from obs_err_new import write_rescalingCoefficient
        # write_rescalingCoefficient('c37rescaling_tropical.dat', \
        #     perc_segs_B, value_segs_B, bases_B, popts_B, \
        #     perc_segs_O, value_segs_O, bases_O, popts_O)
        # exit()

        '''
        Get fit parameters here (offline)
        '''     
        from obs_err_new import rescale_B2O, percentiles2Values, values2Percentiles
        # test_region = 'Global' # can be ['Global', 'Tropical', 'Extra-tropical']
        # print(rescale_B2O([-0.1, 0.0, 0.001, 0.003, 0.01, 0.1, 0.2, 0.3, 0.8, 1.0, 1.1], region=test_region))
        # exit()

        '''
        Plot start here
        '''
        import matplotlib.pyplot as plt
        from collections import OrderedDict

        '''
        Main plot
        '''
        fig = plt.figure(figsize=(7,6))
        ax = plt.gca()
        ax.plot(percentiles, C37_O_perc, color='k')
        ax.plot(percentiles, C37_B_perc, color='r')

        # def expfit(x, a, b, c):
        #     base = (perc_range[0] + perc_range[1]) / 2.0
        #     return a * np.exp(b * (x - base)) + c
        
        # for popt, perc_range, perc_slice in zip(popts_O, perc_ranges_O, perc_slices_O):
        #     ax.plot(percentiles[perc_slice], expfit(percentiles[perc_slice], *popt), color='r', ls='--', lw=0.8)
        # for popt, perc_range, perc_slice in zip(popts_B, perc_ranges_B, perc_slices_B):
        #     ax.plot(percentiles[perc_slice], expfit(percentiles[perc_slice], *popt), color='b', ls='--', lw=0.8)
        ax.plot(percentiles, percentiles2Values(percentiles, perc_segs_O, popts_O, bases_O), color='r', ls='--', lw=0.8)
        ax.plot(percentiles, percentiles2Values(percentiles, perc_segs_B, popts_B, bases_B), color='b', ls='--', lw=0.8)

        ax.set_xlim((0., 100.))
        ax.set_ylim((-0.11, 1.0))
        ax.set_xlabel(r'Percentile [\%]')
        ax.set_ylabel(r'$C37^O / C37^B$')

        cornerText = online_coeff_region + ' [' + self.startSlot + '-' + self.endSlot + ']'
        cornerTag(ax, cornerText, 20.)
        
        '''
        Plot zoom in inset axes of cloud tail
        '''
        axins = ax.inset_axes([0.14, 0.27, 0.6, 0.65])
        axins.plot(percentiles, C37_O_perc, color='k', label=r'$C37^O$')
        axins.plot(percentiles, C37_B_perc, color='r', label=r'$C37^B$')
        
        # for popt, perc_range, perc_slice in zip(popts_O, perc_ranges_O, perc_slices_O):
        #     axins.plot(percentiles[perc_slice], expfit(percentiles[perc_slice], *popt), color='r', ls='--', lw=0.8, label=r'$C37^O$ Fit')
        # for popt, perc_range, perc_slice in zip(popts_B, perc_ranges_B, perc_slices_B):
        #     axins.plot(percentiles[perc_slice], expfit(percentiles[perc_slice], *popt), color='b', ls='--', lw=0.8, label=r'$C37^B$ Fit')
        axins.plot(percentiles, percentiles2Values(percentiles, perc_segs_O, popts_O, bases_O), color='r', ls='--', lw=0.8, label=r'$C37^O$ Fit')
        axins.plot(percentiles, percentiles2Values(percentiles, perc_segs_B, popts_B, bases_B), color='b', ls='--', lw=0.8, label=r'$C37^B$ Fit')
        
        axins.set_xlim(cloud_tail[0], cloud_tail[1])
        axins.set_ylim(cloud_tail[2], cloud_tail[3])
        axins.text(0.05, 0.95, 'Zoom In Of Cloud Tail', ha='left', va='top', size=15, transform=axins.transAxes)
        
        box, cs = ax.indicate_inset_zoom(axins, edgecolor='gray', hatch='/', alpha=1.0)
        box.set_linewidth(1.5)
        for c in cs:
            c.set_linewidth(1.5)
        
        '''
        Legend
        '''
        handles, labels = axins.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        axins.legend(by_label.values(), by_label.keys(), loc='center left', fontsize=15)
        
        plt.savefig(outpic, bbox_inches='tight')
        plt.close()
    
    def plot_c37_pdf(self, ch_no, outpic):
        '''
        Plot c37 Probablity distribution function
        '''
        delta=1e-6
        c37_bins = np.array(list(np.arange(-0.10, -0.04, 0.01)) + \
            list(np.arange(-0.03, 0.20, 0.005)) + list(np.arange(0.20, 1.00, 0.05)))        
        
        dc37_bins = np.diff(c37_bins)

        nc37 = self.get_merged_data(ch_no, 'c37_obser').size
        
        C37_O_pdf, bin_center = self.calc_hist(ch_no, 'c37_obser', c37_bins)
        C37_B_pdf, bin_center = self.calc_hist(ch_no, 'c37_model', c37_bins)
        C37_S_pdf, bin_center = self.calc_hist(ch_no, 'c37_symme', c37_bins)

        C37_O_pdf = C37_O_pdf.astype(np.float32) / nc37
        C37_B_pdf = C37_B_pdf.astype(np.float32) / nc37
        C37_S_pdf = C37_S_pdf.astype(np.float32) / nc37

        C37_O_pdf /= dc37_bins
        C37_B_pdf /= dc37_bins
        C37_S_pdf /= dc37_bins

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7,6))
        ax = plt.gca()

        ax.plot(bin_center, C37_O_pdf, color='r', label=r'$C37^O$', lw=1.5)
        ax.plot(bin_center, C37_B_pdf, color='b', label=r'$C37^B$', lw=1.5)
        ax.plot(bin_center, C37_S_pdf, color='k', label=r'$\overline{C37}$', lw=1.5)

        ax.fill_between(bin_center, C37_O_pdf, C37_B_pdf, where=C37_O_pdf > C37_B_pdf, alpha=0.5, color='r', interpolate=True)
        ax.fill_between(bin_center, C37_O_pdf, C37_B_pdf, where=C37_O_pdf < C37_B_pdf, alpha=0.5, color='b', interpolate=True)

        ax.annotate('Over-estimated Clear Sky', xy=(.0, 1e2), xycoords='data',
            xytext=(0.25, 0.9), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>, head_width=0.2, head_length=0.4', connectionstyle="arc3,rad=0.10", color='k'),
            horizontalalignment='left', verticalalignment='bottom', fontsize=15,
            )
        
        ax.annotate('Under-estimated Thin Cloud', xy=(0.15, 6e-1), xycoords='data',
            xytext=(0.25, 0.6), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>, head_width=0.2, head_length=0.4', connectionstyle="arc3,rad=0.10", color='k'),
            horizontalalignment='left', verticalalignment='bottom', fontsize=15,
            )

        ax.annotate('Under-estimated Thick Cloud', xy=(0.50, 1e-1), xycoords='data',
            xytext=(0.50, 0.45), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>, head_width=0.2, head_length=0.4', connectionstyle="arc3,rad=0.10", color='k'),
            horizontalalignment='left', verticalalignment='bottom', fontsize=15,
            )
        

        ax.fill_between(bin_center, C37_S_pdf, 0, alpha=0.2, color='k')

        ax.set_xlim((bin_center[0], bin_center[-1]))
        ax.set_ylim((1e-3, 1e3))
        ax.set_yscale('log')
        
        ax.legend(loc='upper right')
        ax.set_xlabel(r'C37 [-]')
        ax.set_ylabel(r'Probability Distribution [$dC37^{-1}$]')

        cornerText = 'FY3D MWRI' + ' [' + self.startSlot + '-' + self.endSlot + ']'
        cornerTag(ax, cornerText, 20.)
        
        plt.savefig(outpic, bbox_inches='tight')
        plt.close()

    def get_latband_map(self, ax, slat=None, elat=None):
        '''
        Get Basmap object to plot data on
        '''
        if slat is None:
            slat = self.latS
        if elat is None:
            elat = self.latN

        slon = 0.0
        elon = 360.0

        from mpl_toolkits.basemap import Basemap

        map = Basemap(projection='cyl', llcrnrlat=slat, urcrnrlat=elat, llcrnrlon=slon, urcrnrlon=elon, resolution='l', ax=ax)
        map.drawparallels(np.arange(slat, elat+5, 20.0), linewidth=1, dashes=[4, 3], labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(slon, elon+5, 30.0), linewidth=1, dashes=[4, 3], labels=[0, 0, 0, 1])
        map.drawcoastlines()

        return map
    
    def annot_tropicalband(self, map, ax):
        '''
        Annotate tropical band using box and text
        '''
        for ibox, Tropical_BOX in enumerate(self.Tropical_BOXES):
            latS, latN, lonW, lonE = Tropical_BOX
            lats = [latS, latN, latN, latS, latS]
            lons = [lonW, lonW, lonE, lonE, lonW]
            x, y = map(lons, lats)
            map.plot(x, y, marker=None, color='r', lw=4.0, alpha=0.6)
        
            x, y = map((lonW+lonE)/2.0, (latS+latN)/2.0)

            ax.text(x, y, r'$\textbf{Tropical'+ '[{}]'.format(ibox) +'}$', fontsize=25,
                            ha='center', va='center', color='k')
        
        x, y = map(200., -20.)
        ax.text(x, y, r'$\textbf{ExtraTropical}$', fontsize=25,
                        ha='center', va='center', color='k') 

    def plot_OMB(self, ch_no, outpic):
        '''
        plot OMB of a given channel [No. ch_no]
        '''
        OMB_RAW = self.get_merged_data(ch_no, 'OMB_RAW_sct')
        OMB_BC  = self.get_merged_data(ch_no, 'OMB_BC_sct' )
        OMB_DRT = self.get_merged_data(ch_no, 'OMB_BC_drt')

        bias_raw, std_raw , rmse_raw = self.calc_meanStdRmse(ch_no, 'OMB_RAW_sct')
        bias_bc , std_bc  , rmse_bc  = self.calc_meanStdRmse(ch_no, 'OMB_BC_sct')
        bias_drt, std_drt , rmse_drt = self.calc_meanStdRmse(ch_no, 'OMB_BC_drt')

        tag_raw  = 'OMB RAW'
        tag_bc   = 'OMB BC'
        tag_drt  = 'OMB CLEAR' 

        label_raw = 'Bias={:6>+.3f}K RMSE={:6>.3f}K {}'.format(bias_raw, rmse_raw, tag_raw)
        label_bc  = 'Bias={:6>+.3f}K RMSE={:6>.3f}K {}'.format(bias_bc, rmse_bc, tag_bc)
        label_drt = 'Bias={:6>+.3f}K RMSE={:6>.3f}K {}'.format(bias_drt, rmse_drt, tag_drt)


        bins = np.arange(-30, 30, 1)

        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()

        ax.hist(OMB_RAW, bins=bins, ec="None", fc="blue", alpha=0.3, lw=0.5, label=label_raw)
        ax.hist(OMB_BC, bins=bins,  ec="None",  fc="red", alpha=0.3, lw=0.5, label=label_bc)
        ax.hist(OMB_DRT, bins=bins,  ec="None",  fc="k", alpha=0.3, lw=0.5, label=label_drt)


        plt.plot(bins, gaussian(bins, bias_bc, std_bc, len(OMB_BC)), color='k', ls='--', 
            label='gaussian Distribution of OMB BC')
        # ax.set_yscale('log')
        
        # expand top to place legend
        ylim = ax.get_ylim()
        ax.set_ylim((ylim[0], ylim[1]*1.3))

        cornerText = get_channel_str(self.instrument, ch_no)
        cornerTag(ax, cornerText, 20.)
        
        ax.vlines(x=0.0, ymin=ylim[0], ymax=ylim[1], colors='r', lw=1.5)
        
        plt.xlabel(r'Observation - Simulation [K]')
        plt.ylabel(r'Number of Pixels in Bins [-]')

        rcParams.update({'text.usetex': False, 'font.family':'monospace'})
        plt.legend(fontsize=12, loc='upper left')
        rcParams.update({'text.usetex': True, 'font.family':'serif'})
        plt.tight_layout()
        plt.savefig(outpic)
        plt.close()

    def plot_OVB(self, ch_no, outpic):
        '''
        plot O VS B of a given channel [No. ch_no]
        '''
        O = self.get_merged_data(ch_no, 'O')
        BC = self.get_merged_data(ch_no, 'BC')
        O_BC = O - BC
        B_SCT  = self.get_merged_data(ch_no, 'B_sct')
        B_DRT  = self.get_merged_data(ch_no, 'B_drt')


        ign_perc = 0.2
        xmin, xmax = get_vminvmax(O_BC, vstage=None, ign_perc=ign_perc, lsymmetric=False)
        ymin, ymax = get_vminvmax([B_SCT, B_DRT], vstage=None, ign_perc=ign_perc, lsymmetric=False)
        MIN = min(xmin, ymin)
        MAX = max(xmax, ymax)

        
        '''
        Perform linear regression
        '''
        result = stats.linregress(O_BC, y=B_SCT)
        slope_sct,intercept_sct,rvalue,pvalue,stderr = result
        result = stats.linregress(O_BC, y=B_DRT)
        slope_drt,intercept_drt,rvalue,pvalue,stderr = result

        tag_cloud   = 'CLOUD'
        tag_clear   = 'CLEAR' 
        
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        fig = plt.figure(figsize=(6,6))

        ax = plt.gca()

        ax.scatter(O_BC, B_SCT, s=1., c='r', marker='D', label=tag_cloud, lw=0.)
        ax.scatter(O_BC, B_DRT, s=1., c='k', marker='D', label=tag_clear, lw=0.)

        ax.plot([MIN,MAX],[MIN,MAX], color='blue')
        ax.plot([MIN,MAX],[slope_sct*MIN+intercept_sct, slope_sct*MAX+intercept_sct], \
            ls='--', color='r', label=tag_cloud+' regression')
        ax.plot([MIN,MAX],[slope_drt*MIN+intercept_drt, slope_drt*MAX+intercept_drt], \
            ls='--', color='k', label=tag_clear+' regression')
        

        plt.xlabel(r'Observation [K]')
        plt.ylabel(r'Simulation [K]')

        cornerText = get_channel_str(self.instrument, ch_no)
        cornerTag(ax, cornerText, 20.)
        
        ax.set_xlim((MIN, MAX))
        ax.set_ylim((MIN, MAX))

        rcParams.update({'text.usetex': False, 'font.family':'monospace'})
        plt.legend(fontsize=12, loc='upper left', markerscale=6)
        rcParams.update({'text.usetex': True, 'font.family':'serif'})

        plt.tight_layout()
        plt.savefig(outpic)
        plt.close()

    def plot_OMB_spacedist(self, ch_no, outpic, OMBkey='OMB_BC_sct', dlatbin=2.0, dlonbin=2.0):
        '''
        plot the space distribution of a variable named as OMBkey
        '''
        assert OMBkey in self.OMBkeys or OMBkey in self.normkeys, \
            "Invalid OMBkeys '{}'!".format(OMBkey)

        data = self.get_merged_data(ch_no, OMBkey)
        latTag = self.get_merged_data(ch_no, 'lat')
        lonTag = self.get_merged_data(ch_no, 'lon') 

        pooledData, latCenter, lonCenter = self.pool_data(data, latTag, lonTag, dlatbin, dlonbin, \
            reduceMethod='mean', returnCenter=True)
        
        vstage = 0.5 if OMBkey in self.normkeys else 2.0 # for normed OMB
        dradMIN, dradMAX = get_vminvmax(pooledData, vstage=vstage, ign_perc=2.0, lsymmetric=True)
        print('CH{}: dradMIN={}, dradMAX={}'.format(ch_no, dradMIN, dradMAX))
        # return
    
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import copy

        figsize = (14,6.0)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        map = self.get_latband_map(ax)

        TLAT, TLON = np.meshgrid(latCenter, lonCenter)
        x, y = map(TLON.T, TLAT.T)

        norm = colors.Normalize(vmin=dradMIN, vmax=dradMAX)
        cmap = copy.copy(plt.cm.get_cmap("jet"))

        CF = map.pcolormesh(x, y, pooledData, cmap=cmap, norm=norm)
        CB = fig.colorbar(CF, ax=ax, shrink=0.95, pad=0.10, orientation='horizontal', aspect=60)

        map.fillcontinents(color='white')

        CB.ax.set_xlabel('Pooled Mean First Guess Departure [K]')

        cornerText = get_channel_str(self.instrument, ch_no) + ' [' + self.startSlot + '-' + self.endSlot + ']'
        cornerTag(ax, cornerText, 20.)

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()
    
    def plot_c37_spacedist(self, ch_no, outpic, dlatbin=2.0, dlonbin=2.0):
        '''
        Plot c37 space distribution of C37^O, C37^B, C37^O - C37^B
        '''
        C37_O = self.get_merged_data(ch_no, 'c37_obser')
        C37_B = self.get_merged_data(ch_no, 'c37_model')
        C37_S = self.get_merged_data(ch_no, 'c37_symme')
        
        latTag = self.get_merged_data(ch_no, 'lat')
        lonTag = self.get_merged_data(ch_no, 'lon')

        pooled_C37O, latCenter, lonCenter = self.pool_data(C37_O, latTag, lonTag, dlatbin, dlonbin, \
            reduceMethod='mean', returnCenter=True)

        pooled_C37B, latCenter, lonCenter = self.pool_data(C37_B, latTag, lonTag, dlatbin, dlonbin, \
            reduceMethod='mean', returnCenter=True)

        pooled_dC37 = pooled_C37O - pooled_C37B 

        c37MIN, c37MAX   = get_vminvmax([pooled_C37O, pooled_C37B], vstage=0.02, ign_perc=2.0, lsymmetric=False)
        dc37MIN, dc37MAX = get_vminvmax(pooled_dC37, vstage=0.02, ign_perc=2.0, lsymmetric=True)

        # c37MAX += 0.04
        
        print('c37MIN={}, c37MAX={}'.format(c37MIN, c37MAX))
        print('dc37MIN={}, dc37MAX={}'.format(dc37MIN, dc37MAX))
        # return
        
        '''
        Start plot
        '''
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import copy

        figsize = (20,3.2)
        shrink = 0.95

        TLAT, TLON = np.meshgrid(latCenter, lonCenter)
        cmap = copy.copy(plt.cm.get_cmap("jet"))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(figsize[0],figsize[1]*4.5))

        '''
        1. Observation C37^O
        '''
        norm = colors.Normalize(vmin=c37MIN, vmax=c37MAX)

        map = self.get_latband_map(axes[0])
        x, y = map(TLON.T, TLAT.T)

        CF = map.pcolormesh(x, y, pooled_C37O, cmap=cmap, norm=norm)
        map.fillcontinents(color='white')
        CB = fig.colorbar(CF, ax=axes[0], shrink=shrink, pad=0.01)
        
        axes[0].set_title(r'$\textbf{a) $C37^O$}$')
        self.annot_tropicalband(map, axes[0])

        cornerText = 'FY3D MWRI' + ' [' + self.startSlot + '-' + self.endSlot + ']'
        cornerTag(axes[0], cornerText, 25)
        
        '''
        2. Simulation C37^B
        '''
        map = self.get_latband_map(axes[1])
        x, y = map(TLON.T, TLAT.T)

        CF = map.pcolormesh(x, y, pooled_C37B, cmap=cmap, norm=norm)
        map.fillcontinents(color='white')
        CB = fig.colorbar(CF, ax=axes[1], shrink=shrink, pad=0.01)
        
        axes[1].set_title(r'$\textbf{b) $C37^B$}$')
        self.annot_tropicalband(map, axes[1])

        '''
        3. Obervation - Simulation C37^O - C37^B
        '''
        norm = colors.Normalize(vmin=dc37MIN, vmax=dc37MAX)

        map = self.get_latband_map(axes[2])
        x, y = map(TLON.T, TLAT.T)

        CF = map.pcolormesh(x, y, pooled_dC37, cmap=cmap, norm=norm)
        map.fillcontinents(color='white')
        CB = fig.colorbar(CF, ax=axes[2], shrink=shrink, pad=0.01)
        
        axes[2].set_title(r'$\textbf{c) $C37^O - C37^B$ }$')
        self.annot_tropicalband(map, axes[2])

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()
    
    def plot_OMB_scatter(self, ch_no, outpic, OMBkey='OMB_BC_sct'):
        '''
        Plot the scatter of OMB
        '''
        assert OMBkey in self.OMBkeys or OMBkey in self.normkeys, \
            "Invalid OMBkeys '{}'!".format(OMBkey)
            
        O_RAW = self.get_merged_data(ch_no, 'O')
        BC = self.get_merged_data(ch_no, 'BC')
        B = self.get_merged_data(ch_no, 'B_sct')
        OMB = self.get_merged_data(ch_no, OMBkey)
        latTag = self.get_merged_data(ch_no, 'lat')
        lonTag = self.get_merged_data(ch_no, 'lon') 

        '''
        Add bias on observation
        '''
        O = O_RAW - BC

        figsize = (20,3.2)
        shrink = 0.95

        '''
        Determine vmin, vmax
        '''
        radMIN, radMAX = get_vminvmax([O,B], vstage=10., ign_perc=0.5, lsymmetric=False)
        vstage = 0.5 if OMBkey in self.normkeys else 2.0 # for normed OMB
        dradMIN, dradMAX = get_vminvmax(OMB, vstage=vstage, ign_perc=3.0, lsymmetric=True)
        print('CH{}: radMIN={}, radMAX={}, dradMIN={}, dradMAX={}'.format(ch_no, 
            radMIN, radMAX, dradMIN, dradMAX))
        # return
        
        
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(figsize[0],figsize[1]*4.5))

        '''
        1. Observation
        '''
        map = self.get_latband_map(axes[0])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=O, cmap='rainbow', vmin = radMIN, vmax = radMAX)

        axes[0].set_title(r'$\textbf{a)' + get_channel_str(self.instrument, ch_no) + ' Observation (-Bias)}$')
        
        cb = fig.colorbar(im, ax=axes[0], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('Observed Brightness Temperature [K]')
        
        '''
        2. Simulation
        '''
        map = self.get_latband_map(axes[1])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=B, cmap='rainbow', vmin = radMIN, vmax = radMAX)
        axes[1].set_title(r'$\textbf{b)' + get_channel_str(self.instrument, ch_no) + ' Simulation}$')
        
        cb = fig.colorbar(im, ax=axes[1], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('Simulated Brightness Temperature [K]')

        '''
        3. Obervation - Simulation
        '''
        map = self.get_latband_map(axes[2])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=OMB, cmap='rainbow', vmin = dradMIN, vmax = dradMAX)
        axes[2].set_title(r'$\textbf{c)' + get_channel_str(self.instrument, ch_no) + ' First Guess Departure}$')
        
        cb = fig.colorbar(im, ax=axes[2], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('First Guess Depature [K]')

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()

    def plot_c37_scatter(self, ch_no, outpic):
        '''
        Plot the scatter of C37
        '''
        C37_O = self.get_merged_data(ch_no, 'c37_obser')
        C37_B = self.get_merged_data(ch_no, 'c37_model')
        C37_S = self.get_merged_data(ch_no, 'c37_symme')
        latTag = self.get_merged_data(ch_no, 'lat')
        lonTag = self.get_merged_data(ch_no, 'lon') 

        figsize = (20,3.2)
        shrink = 0.95
        
        # return
        
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(figsize[0],figsize[1]*4.5))

        '''
        1. Observation
        '''
        map = self.get_latband_map(axes[0])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=C37_O, cmap='rainbow', vmin = -0.1, vmax = 0.5)

        axes[0].set_title(r'$\textbf{a)' + get_channel_str(self.instrument, ch_no) + ' Observation Cloud Cover}$')
        
        cb = fig.colorbar(im, ax=axes[0], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('Cloud Indicator [-]')
        
        '''
        2. Simulation
        '''
        map = self.get_latband_map(axes[1])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=C37_B, cmap='rainbow', vmin = -0.1, vmax = 0.5)
        axes[1].set_title(r'$\textbf{b)' + get_channel_str(self.instrument, ch_no) + ' Simulation Cloud Cover}$')
        
        cb = fig.colorbar(im, ax=axes[1], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('Cloud Indicator [-]')

        '''
        3. Obervation - Simulation
        '''
        map = self.get_latband_map(axes[2])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=C37_S, cmap='rainbow', vmin = -0.1, vmax = 0.5)
        axes[2].set_title(r'$\textbf{c)' + get_channel_str(self.instrument, ch_no) + ' Symmetric Cloud Cover}$')
        
        cb = fig.colorbar(im, ax=axes[2], shrink=shrink, pad=0.01)
        cb.ax.set_ylabel('Cloud Indicator [-]')

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()
    
    def plot_normalized_OMB(self, ch_no, outpic, OMBkey='OMB_BC_sct', 
            c37keys=['c37_symme*'], c37Colors=['r'], c37Labels=[r'$\overline{C37}$']):
        '''
        Plot the distribution of normalized OMB
        '''
        assert OMBkey in self.OMBkeys, \
            "Invalid OMBkey '{}'!".format(OMBkey)
        
        for c37key in c37keys:
            assert c37key in self.c37keys, \
                "Invalid c37key '{}'!".format(c37key)

        # Get OMB normalized by different c37keys  
        normalized_OMBs = []
        normkeys = []
        for c37key in c37keys:
            normkey = self.get_normed_key(OMBkey, c37key) 
            assert normkey in self.normkeys, \
                "Invalid normkey '{}'!".format(normkey)
            normalized_OMB = self.get_merged_data(ch_no, normkey)
            normalized_OMBs.append(normalized_OMB)
            normkeys.append(normkey)
        
        # Get alltogether normalized OMB 
        OMB = self.get_merged_data(ch_no, OMBkey)
        mean, std, rmse = self.calc_meanStdRmse(ch_no, OMBkey)
        normalized_OMBs.append(OMB/std)

        # Get bins by getvminmax
        dradMIN, dradMAX = get_vminvmax(normalized_OMB, vstage=0.5, ign_perc=0.1, lsymmetric=True)
        dradMIN, dradMAX = -10., 10.
        print('CH{}: dradMIN={:>.2f}, dradMAX={:>.2f}'.format(ch_no, dradMIN, dradMAX))

        drad_bins_gaussian = np.linspace(dradMIN, dradMAX, 101)
        drad_bins = np.array( \
            list(np.arange(dradMIN, -2.0, 1.0)) + \
            list(np.arange(-2.0, 2.0, 0.10)) + \
            list(np.arange(2.0, dradMAX, 1.0))) 

        '''
        Start plot here
        '''
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()

        hist_alltogether, binCenter = self.calc_hist(ch_no, OMBkey, drad_bins, norm=std)
        pdf_alltogether = hist_alltogether / np.sum(hist_alltogether) / np.diff(drad_bins)
        ax.plot(binCenter, pdf_alltogether, color='k', ls='-', label='Nomalized Alltogether')

        # Get gaussian distribution
        gauss = gaussian(drad_bins_gaussian, mean/std, 1.0, 1/(drad_bins_gaussian[1]-drad_bins_gaussian[0]))
        ax.plot(drad_bins_gaussian, gauss, color='k', ls='--', label='gaussian Distribution')

        for c37Color, c37Label, normkey in zip(c37Colors, c37Labels, normkeys):
            hist_normalized, binCenter = self.calc_hist(ch_no, normkey, drad_bins)
            pdf_normalized = hist_normalized / np.sum(hist_normalized) / np.diff(drad_bins)
            ax.plot(binCenter, pdf_normalized, color=c37Color, ls='-', label='Normalized by ' + c37Label)        
        
        cornerText = get_channel_str(self.instrument, ch_no)
        cornerTag(ax, cornerText, 20.)
        
        ax.set_ylim([1e-4, 2e0])
        ax.set_yscale('log')

        plt.xlabel(r'Normalized FG departure')
        plt.ylabel(r'PDF')

        plt.legend(fontsize=12, loc='upper left')
        plt.tight_layout()
        plt.savefig(outpic)
        plt.close()

    def plot_BgQC(self, ch_no, outpic, OMBkey='OMB_BC_sct', c37key='c37_symme*', thresh_sigma=2.5):
        '''
        Look into BgQC
        '''

        # 1. Check input keys
        assert OMBkey in self.OMBkeys, \
            "Invalid OMBkey '{}'!".format(OMBkey)
        
        assert c37key in self.c37keys, \
            "Invalid c37key '{}'!".format(c37key)

        normkey = self.get_normed_key(OMBkey, c37key) 
        
        assert normkey in self.normkeys, \
                "Invalid normkey '{}'!".format(normkey)


        # 2. Get data
        latTag = self.get_merged_data(ch_no, 'lat')
        lonTag = self.get_merged_data(ch_no, 'lon')

        OMB = self.get_merged_data(ch_no, OMBkey)
        OMBnorm = self.get_merged_data(ch_no, normkey)
        BgQCFlag = (OMBnorm>thresh_sigma) | (OMBnorm<-thresh_sigma)
        print('ch{}: BgQC screened out {} Pixels'.format(ch_no, np.sum(BgQCFlag)))
        OMBnormBgQC = np.where(BgQCFlag, np.nan, OMBnorm)

        dradMIN, dradMAX = get_vminvmax(OMB, vstage=2.0, ign_perc=2.0, lsymmetric=True)
        dradnormMIN, dradnormMAX = -thresh_sigma-0.5, thresh_sigma+0.5

        # 3. Plot start here
        import matplotlib.pyplot as plt
        
        figsize = (20,3.2)
        shrink = 0.95

        TitleFontSize=22

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(figsize[0],figsize[1]*4.5))

        map = self.get_latband_map(axes[0])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=OMB, cmap='rainbow', vmin = dradMIN, vmax = dradMAX)

        axes[0].set_title(r'$\textbf{a)' + get_channel_str(self.instrument, ch_no) + ' First Guess Departure [K]}$', fontsize=TitleFontSize)
        cb = fig.colorbar(im, ax=axes[0], shrink=shrink, pad=0.01)

        map = self.get_latband_map(axes[1])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=OMBnorm, cmap='rainbow', vmin = dradnormMIN, vmax = dradnormMAX)

        axes[1].set_title(r'$\textbf{b)' + get_channel_str(self.instrument, ch_no) + ' Normalized First Guess Departure }$', fontsize=TitleFontSize)
        cb = fig.colorbar(im, ax=axes[1], shrink=shrink, pad=0.01)
        
        map = self.get_latband_map(axes[2])
        x, y = map(lonTag, latTag)
        im = map.scatter(x, y, marker='D', s=3, c=OMBnormBgQC, cmap='rainbow', vmin = dradnormMIN, vmax = dradnormMAX)
        map.scatter(x[BgQCFlag], y[BgQCFlag], marker='1', s=5, c='k')

        cornerText = r'$\sigma_t={'+ '{:>.2f}'.format(thresh_sigma) +'}$'
        cornerTag(axes[2], cornerText, 25)
        
        axes[2].set_title(r'$\textbf{c)' + get_channel_str(self.instrument, ch_no) + ' Normalized First Guess Departure + BgQC}$', fontsize=TitleFontSize)
        cb = fig.colorbar(im, ax=axes[2], shrink=shrink, pad=0.01)

        plt.savefig(outpic, bbox_inches='tight')
        plt.close()
    