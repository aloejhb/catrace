import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def read_trace_file(trace_file):
    time_trace = sio.loadmat(trace_file)
    df_trace_mat = np.stack(time_trace['timeTraceDfMatList'][0])
    odor_list = np.stack(time_trace['odorList'][0]).squeeze()
    return df_trace_mat, odor_list


def plot_response_pca(pattern, odor_list, n_trial, ax, scatterkwargs={}, ellipsekwargs={}, fig_title=''):
    n_odor = len(odor_list)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(pattern)
    color_seq = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i in range(n_odor):
        # if i == 0:
        #     pcx = pc[0:2, :]
        # else:
        #     pcx = pc[i*n_trial-1:(i+1)*n_trial-1, :]
        pcx = pc[i*n_trial:(i+1)*n_trial, :]
        color = color_seq[i]
        ax.scatter(pcx[:, 0], pcx[:, 1], c=color, label=odor_list[i], **scatterkwargs)
        # ax.legend(framealpha=0.5)
        confidence_ellipse(pcx[:, 0], pcx[:, 1], ax, n_std=2, edgecolor=color, alpha=0.5, **ellipsekwargs)
    if fig_title:
        ax.set_title(fig_title)
        

if __name__ == '__main__':
    pass
    # data_root_dir = '/home/hubo/Projects/Ca_imaging/results/'
    # dp_exp = '2019-09-03-OBfastZ'
    # ob_exp = '2019-09-03-OBFastZ2'

    # ob_dir = os.path.join(data_root_dir,ob_exp,'time_trace')
    # dp_dir = os.path.join(data_root_dir,dp_exp,'time_trace')

    # ob_df_dict = {}
    # for plane_num in range(2,5):
    #     plane_dir = 'plane{0:02d}'.format(plane_num)
    #     time_window = np.array([14, 19])
    #     trace_file = os.path.join(ob_dir,plane_dir,'timetrace.mat')
    #     df, ol = read_trace_file(trace_file)
    #     ob_df_dict[plane_num] = df
    #     fig_title = 'OB plane {0:d}'.format(plane_num)
    #     plot_response_pca(df, ol, time_window, fig_title=fig_title)

    # ob_df_trace_mat = np.concatenate((ob_df_dict[2], ob_df_dict[3], ob_df_dict[3]), axis=1)
    # plot_response_pca(ob_df_trace_mat, ol, time_window, fig_title='OB')
    
    

    # df_dict = {}
    # odor_list_dict = {}
    # for plane_num in range(1, 3):
    #     plane_dir = 'plane{0:02d}'.format(plane_num)
    #     trace_file = os.path.join(dp_dir, plane_dir, 'timetrace.mat')
    #     df, ol = read_trace_file(trace_file)
    #     df_dict[plane_num] = df
    #     odor_list_dict[plane_num] = ol

    # df_trace_mat = np.concatenate((df_dict[1], df_dict[2]), axis=1)
    # time_window = np.array([13.3, 15.5])
    # plot_response_pca(df_trace_mat, odor_list_dict[1], time_window, fig_title='Dp')
    # plt.show()
