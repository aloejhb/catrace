import numpy as np
import rcca

def compute_cca(dimred_list, n_components):
    cca = rcca.CCA(kernelcca = False, reg = 0., numCC = n_components)
    data_list = [dimred['latent'] for dimred in dimred_list]
    index_list = [dimred['index'] for dimred in dimred_list]
    cca.train(data_list)
    ccacomp_list = [dict(latent=comps, index=index_list[idx]) for idx,comps in enumerate(cca.comps)]
    return ccacomp_list


def generate_random_latent(source_dimred):
    pp = source_dimred['latent']
    randommat = np.random.normal(scale=pp.std(), size=pp.shape)
    return dict(latent=randommat, index=source_dimred['index'])
