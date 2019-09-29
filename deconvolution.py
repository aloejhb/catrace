#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of how to use the program for deconvolving your data.

The demo is using a small dataset of 110 neurons, recorded simultaneously in a single FOV.
Recording rate: 28 Hz (resonant scanning).
Brain area: area Dp (piriform cortex homolog) and area Dl (hippocampal homolog) in adult zebrafish (brain explant, room temperature).
Calcium indicator: GCaMP6f, expressed using a NeuroD promotor fragment (https://www.osapublishing.org/boe/abstract.cfm?uri=boe-7-5-1656 for details).
Recording duration: ca. 3 minutes.
Only spontaneous activity.

@author: Peter Rupprecht, peter.rupprecht@fmi.ch
"""

from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
sys.path.append('../Spikefinder-Elephant/')
from elephant.utils2 import extract_stats, genhurst, map_between_spaces
from elephant.utils import norm
from elephant.c2s_preprocessing import preprocess, percentile_filter
from copy import deepcopy
from sklearn.decomposition import PCA
from importlib import reload

logging.basicConfig(filename='deconv.log', format='%(asctime)s - %(message)s', level=logging.INFO)

exec(open("../Spikefinder-Elephant/elephant/config_elephant.py").read())
exec(open("../Spikefinder-Elephant/elephant/2_model.py").read())
model.load_weights("../Spikefinder-Elephant/models/model1.h5")

def read_trace_file(trace_file):
    time_trace = sio.loadmat(trace_file)
    df_trace_mat = np.stack(time_trace['timeTraceDfMatList'][0])
    odor_list = np.stack(time_trace['odorList'][0]).squeeze()
    return df_trace_mat, odor_list


def predict_spike(trace, fs):
    
    tracex, fsx = preprocess(trace, fs)
    Ypredict = np.zeros(tracex.shape) * np.nan  # with nan padding
    for k in range(0, trace.shape[1]):
        if not (k+1) % 10:
            logging.info('Predicting spikes for neuron %s out of %s' % (k+1, trace.shape[1]))
        x1x = tracex[:, k]
        idx = ~np.isnan(x1x)
        calcium_traceX = norm(x1x[idx])
        # initialize the prediction vector
        XX = np.zeros((calcium_traceX.shape[0]-windowsize,windowsize,1), dtype=np.float32)
        for jj in range(0,(calcium_traceX.shape[0]-windowsize)):
            XX[jj,:,0] = calcium_traceX[jj:(jj+windowsize)]
        A = model.predict( XX,batch_size = 4096 )
        indices = slice( int(windowsize*before_frac), int(windowsize*before_frac+len(A)) )
        Ypredict[ indices,k ] = A[:,0]
    return Ypredict


def read_and_predict(root_dir, plane_num, fs):
    start_frame = int(8 * fs)
    plane_dir = 'plane{0:02d}'.format(plane_num)
    trace_file = os.path.join(root_dir,'time_trace',plane_dir,'timetrace.mat')
    df, ol = read_trace_file(trace_file)
    logging.info(trace_file)
    for i in range(len(df)):
        logging.info('Trial no. {0:03d}'.format(i))
        trace = np.transpose(df[i][:, start_frame:])
        spike_prediction = predict_spike(trace, fs)
        spike_dir = os.path.join(root_dir, 'deconvolution', plane_dir)
        if not os.path.exists(spike_dir):
            os.makedirs(spike_dir)
        spike_file = os.path.join(spike_dir, 'spike_{0:03d}.npy'.format(i))
        np.save(spike_file, spike_prediction)


if __name__ == '__main__':
    data_root_dir = '/home/hubo/Projects/Ca_imaging/results/'
    dp_exp = '2019-09-03-OBfastZ'
    ob_exp = '2019-09-03-OBFastZ2'

    ob_dir = os.path.join(data_root_dir, ob_exp)
    dp_dir = os.path.join(data_root_dir, dp_exp)

    fs = 30/4
    for plane_num in range(2,5):
        read_and_predict(ob_dir, plane_num, fs)


    for plane_num in range(1, 3):
        read_and_predict(dp_dir, plane_num, fs)
    
