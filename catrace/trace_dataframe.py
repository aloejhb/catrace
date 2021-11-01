import numpy as np


def get_colname(coltype, plane_nb):
    return '{0:s}_plane{1:02d}'.format(coltype, plane_nb)


def concatenate_planes(tracedf, plane_nb_list):
    trace_list = [None] * len(tracedf)
    for i in range(len(tracedf)):
        trace = [tracedf[get_colname('dfovf', k)][i] for k in plane_nb_list]
        trace = np.concatenate(trace)
        trace_list[i] = trace
    return trace_list


