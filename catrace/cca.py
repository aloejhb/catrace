import numpy as np
import rcca

from .dim_reduce import get_embeddf


def compute_cca(embeddf_list, n_components):
    cca = rcca.CCA(kernelcca = False, reg = 0., numCC = n_components)
    data_list = [df.to_numpy() for df in embeddf_list]
    cca.train(data_list)
    ccacomp_list = [get_embeddf(comps, embeddf_list[i].index) for i,comps in enumerate(cca.comps)]
    return ccacomp_list


def generate_random_latent(source_dimred):
    pp = source_dimred['latent']
    randommat = np.random.normal(scale=pp.std(), size=pp.shape)
    return dict(latent=randommat, index=source_dimred['index'])
