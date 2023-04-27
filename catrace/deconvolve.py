import os
import sys
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml

from cascade2p import cascade # local folder


noise_levels = plot_noise_level_distribution(traces, frame_rate)
spike_prob = cascade.predict(model_name, traces)
