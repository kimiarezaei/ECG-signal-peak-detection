
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:07:05 2022

@author: KimiaR
"""


import numpy as np
from scipy import signal        #signal processing library of python
import wfdb                   #python waveform-database (WFDB) package. A library of tools for reading, writing, and processing WFDB signals and annotations.
from scipy.signal import find_peaks    #peak detection
import glob                              #reading data from folders
from wfdb import processing             #evaluation

### defining parameters
Fs = 360                          #sampling frequncy of data

#High Pass Filter design
deltaF = 2                             # Transition Bandwidth in Hz
passbandF=7                            # Passband Edge Frequency in Hz
M=int(np.ceil(3.1 * Fs/deltaF))           # Filter Length  ******Check Odd******
M=M+1                                    #filter length should be odd
MidM = int((M-1)/2)                     #Filter Midpoint
Fc = passbandF-(deltaF/2)                          # Cutoff Frequency in Hz
ncoeffHP = signal.firwin(M, Fc, window = 'hann', fs = Fs, pass_zero=False) 

#Lowpass Filter design 
deltaLPF = 6                                        # Transition Bandwidth in Hz
LPPassbandF = 12                                    # Passband Edge Frequency in Hz
MLP=int(np.ceil(3.1 * Fs/deltaLPF))                 # Filter Length  ******Check Odd******
MLP=MLP+1                                           #filter length should be odd#

MidMLP = int((MLP-1)/2)                             #Filter Midpoint
FcLP= LPPassbandF + (deltaLPF/2)                    # Cutoff Frequency in Hz
ncoeffLP = signal.firwin(MLP, FcLP, window = 'hann', fs = Fs, pass_zero=True)   #hanning window has the best result
 
#moving average window
Nofsamples=int(0.2*Fs)                 #determine the number of sample based on a window of 200ms = 0.2s
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'full') / w

#read signals and annotations
list_of_files = glob.glob('mitdb/*.dat')               # create the list of signals
list_of_files_ann = glob.glob('mitdb/*.atr')           # create the list of annotaions

count_sample = 0

#preprocessing and peak detection 
peaksN_arr = np.array([]) 
ann_arr = np.array([])
TP_arr = np.array([])
FN_arr = np.array([])
FP_arr = np.array([])
Se_arr = np.array([])
plusP_arr = np.array([])

for i in range(len(list_of_files)):
    count_sample += 1
    
    #read annotaions
    file_name_ann=list_of_files_ann[i].replace('.atr','')
    annotation = wfdb.rdann(file_name_ann, 'atr')
    annotation_sample=annotation.sample
    annN=len(annotation_sample)                    #number of peaks (based on annotation file)
    ann_arr=np.append(ann_arr, annN)
    
    
    #read ECG signals
    file_name_siganl=list_of_files[i].replace('.dat','')
    ecg_data, fields = wfdb.rdsamp(file_name_siganl, channels =[0]) 
    ecg_data=ecg_data.reshape(1,-1)
    ecg_data=np.array(ecg_data[0])
    
    #highpass filter
    ecg_HPoutput = signal.lfilter(ncoeffHP, [1.0], ecg_data)
    ecg_HPoutputLR = ecg_HPoutput[MidM:len(ecg_HPoutput)] 
    
    #lowpass filter
    ecg_LPoutput = signal.lfilter(ncoeffLP, [1.0], ecg_HPoutputLR)   
    ecg_LPoutputLR = ecg_LPoutput[MidMLP:len(ecg_LPoutput)]
    
    
    #Other filters
    ecg_derivative=np.diff(ecg_LPoutputLR) 
    ecg_squared=ecg_derivative**2
    ecg_movingaverage=moving_average(ecg_squared,Nofsamples)
    
    
    #check the signal distributuion to find the range of most peaks
    Nbins=20                            #number of bins in histogram 
    counts, bins = np.histogram(ecg_movingaverage, bins=Nbins)
    max_height=np.mean(np.diff(bins))               #the height(range for peaks in ecg signal) which should be used in peak detection
        
    
    
    #peak detection
    peaks = find_peaks(ecg_movingaverage, height=max_height, distance=Nofsamples)[0]
    
    
    #method benchmarking
    comparitor = processing.Comparitor(annotation_sample, peaks, Nofsamples)     #compare annotations with detected peaks                           
    comparitor.compare() 
    FN=comparitor.fn                               #number of false negatives
    FP=comparitor.fp                               #number of false positives
    Se=comparitor.sensitivity                      #calculating sensitivity
    plusP=comparitor.positive_predictivity         #calculating positive predictivity
    
    
    #condition for signals with different range to have a better detection result
    if (Se<0.6 or plusP<0.6) and FN>FP:
        Nbins *=4                       #increasing the number of bins, reduce the bins edge so we parameterize lower height in peak detection
        counts, bins = np.histogram(ecg_movingaverage, bins=int(Nbins))
        max_height=np.mean(np.diff(bins))                 #the height(range for peaks in ecg signal) which should be used in peak detection
        peaks = find_peaks(ecg_movingaverage, height=max_height, distance=Nofsamples)[0]
        
    elif(Se<0.6 or plusP<0.6) and FN<FP:
        Nbins /=4                       #increasing the number of bins, reduce the bins edge so we parameterize lower height in peak detection
        counts, bins = np.histogram(ecg_movingaverage, bins=int(Nbins))
        max_height=np.mean(np.diff(bins))                 #the height(range for peaks in ecg signal) which should be used in peak detection
        peaks = find_peaks(ecg_movingaverage, height=max_height, distance=Nofsamples)[0]
        
        

    comparitor = processing.Comparitor(annotation_sample, peaks, Nofsamples)     #compare annotations with detected peaks                           
    comparitor.compare()
    
    peaksN=len(peaks)                              #number of detected peaks 
    TP=comparitor.tp                               #number of true positives
    FN=comparitor.fn                               #number of false negatives
    FP=comparitor.fp                               #number of false positives
    Se=comparitor.sensitivity                      #calculating sensitivity
    plusP=comparitor.positive_predictivity         #calculating positive predictivity
    #print(file_name_siganl)
    #comparitor.print_summary()
    

    
    peaksN_arr=np.append(peaksN_arr, peaksN)
    TP_arr=np.append(TP_arr, TP)
    FN_arr=np.append(FN_arr, FN)
    FP_arr=np.append(FP_arr, FP)
    Se_arr=np.append(Se_arr, Se)
    plusP_arr=np.append(plusP_arr, plusP)
        
    

totalann=int(np.sum(ann_arr))
totalpeaks=int(np.sum(peaksN_arr))
TPN=int(np.sum(TP_arr))
FNN=int(np.sum(FN_arr))
FPN=int(np.sum(FP_arr))
Se_mean=np.mean(Se_arr)
plusP_mean=np.mean(plusP_arr)
Se_plusP_mean=np.mean([Se_mean,plusP_mean])

Se_min=np.min(Se_arr)
plusP_min=np.min(plusP_arr)
Se_plusP_min=np.min([Se_min,plusP_min])



print('Total peaks(annotation file):', totalann ,'Total detected peaks:', totalpeaks,'TP:', TPN,'FN:',FNN,'FP:',FPN,'Se(%):', "{:.3f}".format(Se_mean*100),'+P(%):',  "{:.3f}".format(plusP_mean*100), 'mean (Se,+P):',"{:.3f}".format(Se_plusP_mean*100) , 'min (Se,+P):',"{:.3f}".format(Se_plusP_min*100))



    

