import sys
import numpy as np
import re
sys.path.append('../')
from catrace import dataio
import catrace.process_time_trace as ptt
import catrace.plot_trace as pltr
import catrace.pattern_correlation as pcr
import catrace.manifold_embed as emb
from catrace.frame_time import convert_sec_to_frame
from catrace.trace_dataframe import concatenate_planes


sys.path.append('/home/hubo/Projects/Ca_imaging/external_packages/Cascade/')
from cascade2p.utils import calculate_noise_levels



def compute_experiment_noise(data_root_dir, exp_name, region, plane_nb_list, offset):
    # Load data
    exp_subdir = os.path.join(exp_name, region)
    # exp_info = dataio.load_experiment(data_root_dir, exp_subdir, exp_name=exp_name)
    # frame_rate = exp_info['frame_rate'] / exp_info['num_plane']
    # plane_nb_list = range(1, 5)
    frame_rate = 30/4
    num_trial = 3
    exp_info = dict(num_trial=3)
    # odor_list = ['ala', 'trp', 'ser', 'tdca', 'tca', 'gca', 'acsf', 'spont']
    odor_list = ['phe', 'trp', 'arg', 'tdca', 'tca', 'gca', 'acsf', 'spont']
    tracedf = dataio.load_trace_file(data_root_dir, exp_subdir, plane_nb_list, num_trial, odor_list)
    # Drop neuron 33 in plane 4, since its dF/F is inf
    # idx = pd.IndexSlice
    # tracedf.drop(tracedf.loc[idx[:, :, 3, 33],:].index, inplace=True)

    # Cut first X second to exclude PMT off period
    cut_time = 5+2
    cut_win = convert_sec_to_frame([cut_time, 40], frame_rate)
    tracedf = ptt.cut_tracedf(tracedf, cut_win[0], 0, cut_win[1])


    # Calculate dF/F
    fzero_twindow = np.array([7, 9])
    dfovf = ptt.compute_dfovf(tracedf, fzero_twindow, frame_rate, intensity_offset=offset)

    quant = 0.9
    print('dfovf {:.2f} quantile: {:.2f}'.format(quant, np.quantile(dfovf, quant)))
    neurons_x_time = dfovf.values
    noise_levels = calculate_noise_levels(neurons_x_time, frame_rate)
    return noise_levels


if __name__ == '__main__':
    data_root_dir = '/home/hubo/Projects/Ca_imaging/results/'

    region = 'OB'
    exp_list = ['2021-07-31-JH17','2021-09-04-JH18']
    # region = 'OB'
    # exp_list = ['2021-02-05-JH9', '2021-03-18-JH10', '2021-03-19-JH10',
    #             '2021-04-02-JH11', '2021-04-03-JH11', '2021-05-01-JH13',
    #             '2021-05-22-JH14']
    # exp_list = '2021-03-19-JH10',
    # exp_list = ['2021-07-15-DpOBEM-N2', '2021-07-16-DpOBEM-N3']
    # exp_list = ['2021-02-05-JH9', '2021-03-18-JH10', '2021-03-19-JH10',
                # '2021-04-02-JH11', '2021-04-03-JH11', '2021-05-01-JH13',
                # '2021-05-22-JH14']
    plane_nb_list = np.array([1,2,3,4]) - 1


    exp_list = [re.sub('-JH', '-DpOBEM-JH', exp_name)
                for exp_name in exp_list]
    offset = -0
    print('offset: {}'.format(offset))
    for exp_name in exp_list:
        noise_levels = compute_experiment_noise(data_root_dir, exp_name,
                                                region, plane_nb_list,
                                                offset)
        print('{}\t{:.2f}'.format(exp_name, np.mean(noise_levels)))
