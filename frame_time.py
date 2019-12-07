import numpy as np


def convert_sec_to_frame(sec, frame_rate):
    return (np.array(sec) * frame_rate).astype('int')
