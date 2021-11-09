import numpy as np

def get_nrn_idx_for_component(comp_idx, wmat, kmat, thresh, criterion):
    qmat = np.dot(kmat.T,wmat)

    if criterion == 'abs':
        nrn_idx = np.abs(qmat[comp_idx,:]) >= thresh
    elif criterion == 'above':
        nrn_idx = qmat[comp_idx,:] >= thresh
    elif criterion == 'below':
        nrn_idx = qmat[comp_idx,:] <= thresh
    else:
        raise ValueError('Criterion should be either abs, above or below')
    return nrn_idx


def get_comp_pattern(pattern, wmat, kmat, comp_idx, thresh, criterion):
    nrn_idx = get_nrn_idx_for_component(comp_idx, wmat, kmat, thresh, criterion)
    comp_pattern = pattern.loc[:,nrn_idx]
    return comp_pattern


def get_comp_pattern_from_list(exp_idx, exp_list, pca_dlist, pattern_dlist, cca,
                               comp_kwargs):
    exp = exp_list[exp_idx]
    pca = pca_dlist[exp]['pca']
    pattern = pattern_dlist[exp]
    wmat = pca.components_
    kmat = cca.ws[exp_idx]
    comp_pattern = get_comp_pattern(pattern, wmat, kmat, **comp_kwargs)
    return comp_pattern


def get_comp_pattern_list(exp_list, pca_dlist, pattern_dlist, cca, comp_kwargs):
    comp_pattern_list = [get_comp_pattern_from_list(k, exp_list, pca_dlist,
                                                    pattern_dlist, cca, comp_kwargs)
                         for k in range(len(exp_list))]
    return comp_pattern_list


# fig, ax = plt.subplots(figsize=(12,10))
# ax.imshow(comp_pattern.T, aspect='auto', interpolation='none')
