import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from .process_time_trace import select_time_points, select_odors_df
from .process_neuron import sample_neuron
from .utils import get_odor_pairs
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LdaCrossValParameters:
    num_neurons: int = None
    sample_neuron_seed: int = None
    k: int = 5
    manifold_level: str = 'manifold'

def lda_crossval(dff, params):
    """
    Perform LDA classification with k-fold cross-validation.

    Parameters
    ----------
    dff : pandas.DataFrame
        Multi-indexed dataframe with (manifold_level, 'trial', 'time') index and neurons as columns.
    k : int, default=5
        Number of folds for cross-validation.

    Returns
    -------
    scores : np.ndarray
        Cross-validation accuracies (one per fold).
    """
    if params.num_neurons is not None:
        # Randomly select num_neurons if specified
        if params.num_neurons > dff.shape[1]:
            raise ValueError("num_neurons exceeds the number of available neurons.")
        dff = dff.sample(n=params.num_neurons, axis=1, random_state=params.sample_neuron_seed)

    # Prepare data
    y = dff.index.get_level_values(params.manifold_level).to_numpy()
    X = dff.to_numpy()

    # Fit LDA with cross-validation
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, X, y, cv=params.k)

    return scores


def lda_manifold_pair(dff, lda_cross_val_params):
    manifold_level = lda_cross_val_params.manifold_level
    # Get all odors
    manifold_names = dff.index.unique(manifold_level)
    manifold_pairs = get_odor_pairs(manifold_names, manifold_names)
    mean_scores = []

    for manifold_pair in manifold_pairs:
        dff_pair = select_odors_df(dff, manifold_pair)

        # Perform LDA with cross-validation
        scores = lda_crossval(dff_pair, lda_cross_val_params)
        mean_scores.append(scores.mean())

    cvdf = pd.DataFrame({'manifold1': [pair[0] for pair in manifold_pairs],
                         'manifold2': [pair[1] for pair in manifold_pairs],
                         'mean_lda_cv_score': mean_scores})
    return cvdf


def sample_and_compute_lda(df, lda_params, manifold_names, time_window, sample_size, seed):
    df = select_odors_df(df, manifold_names)
    df = sample_neuron(df, sample_size=sample_size, seed=seed)
    df = select_time_points(df, time_window)
    cvdf = lda_manifold_pair(df, lda_params)
    return cvdf