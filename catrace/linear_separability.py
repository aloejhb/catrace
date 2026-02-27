import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline  # <<< MODIFIED (MERGE + CHOICE)
from sklearn.preprocessing import StandardScaler  # <<< MODIFIED (MERGE + CHOICE)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # <<< MODIFIED (MERGE + CHOICE)
from sklearn.svm import SVC  # <<< MODIFIED (MERGE + CHOICE)

from .process_time_trace import select_time_points, select_odors_df
from .process_neuron import sample_neuron
from .utils import get_odor_pairs
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ClassifierCrossValParameters:  # <<< MODIFIED (MERGE + CHOICE)
    # Common
    k: int = 5
    manifold_level: str = 'manifold'
    model: str = 'lda'  # <<< MODIFIED (MERGE + CHOICE)  # 'lda' or 'svm'

    # LDA params
    solver: str = 'svd'  # <<< MODIFIED (MERGE + CHOICE)
    shrinkage: float = None  # <<< MODIFIED (MERGE + CHOICE)

    # SVM params
    C: float = 1.0  # <<< MODIFIED (MERGE + CHOICE)
    kernel: str = 'linear'  # <<< MODIFIED (MERGE + CHOICE)
    gamma: str | float = 'scale'  # <<< MODIFIED (MERGE + CHOICE)
    class_weight: str | dict | None = None  # <<< MODIFIED (MERGE + CHOICE)
    scale_for_svm: bool = True  # <<< MODIFIED (MERGE + CHOICE)


def _build_estimator(params: ClassifierCrossValParameters):  # <<< MODIFIED (MERGE + CHOICE)
    """
    Return a sklearn estimator according to params.model.
    """
    model = params.model.lower()  # <<< MODIFIED (MERGE + CHOICE)

    if model == 'lda':  # <<< MODIFIED (MERGE + CHOICE)
        return LinearDiscriminantAnalysis(
            solver=params.solver,
            shrinkage=params.shrinkage
        )

    if model == 'svm':  # <<< MODIFIED (MERGE + CHOICE)
        svm = SVC(
            C=params.C,
            kernel=params.kernel,
            gamma=params.gamma,
            class_weight=params.class_weight
        )
        # Scaling is usually important for SVM
        return make_pipeline(StandardScaler(), svm) if params.scale_for_svm else svm  # <<< MODIFIED (MERGE + CHOICE)

    raise ValueError(f"Unknown model '{params.model}'. Use 'lda' or 'svm'.")  # <<< MODIFIED (MERGE + CHOICE)


def crossval(dff, params: ClassifierCrossValParameters):  # <<< MODIFIED (MERGE + CHOICE)
    """
    Perform classification (LDA or SVM) with k-fold cross-validation.
    """
    y = dff.index.get_level_values(params.manifold_level).to_numpy()
    X = dff.to_numpy()

    est = _build_estimator(params)  # <<< MODIFIED (MERGE + CHOICE)
    scores = cross_val_score(est, X, y, cv=params.k)
    return scores


def manifold_pair_cv(dff, params: ClassifierCrossValParameters):  # <<< MODIFIED (MERGE + CHOICE)
    manifold_level = params.manifold_level  # <<< MODIFIED (MERGE + CHOICE)
    manifold_names = dff.index.unique(manifold_level)
    manifold_pairs = get_odor_pairs(manifold_names, manifold_names)

    mean_scores = []
    for manifold_pair in manifold_pairs:
        dff_pair = select_odors_df(dff, manifold_pair)
        scores = crossval(dff_pair, params)  # <<< MODIFIED (MERGE + CHOICE)
        mean_scores.append(scores.mean())

    score_col = f"mean_{params.model.lower()}_cv_score"  # <<< MODIFIED (MERGE + CHOICE)
    cvdf = pd.DataFrame({
        'manifold1': [pair[0] for pair in manifold_pairs],
        'manifold2': [pair[1] for pair in manifold_pairs],
        score_col: mean_scores  # <<< MODIFIED (MERGE + CHOICE)
    })
    return cvdf


def sample_and_compute_classifier(df, params: ClassifierCrossValParameters, manifold_names, time_window, sample_size, seed):  # <<< MODIFIED (MERGE + CHOICE)
    df = select_odors_df(df, manifold_names)
    df = sample_neuron(df, sample_size=sample_size, seed=seed)
    df = select_time_points(df, time_window)
    cvdf = manifold_pair_cv(df, params)  # <<< MODIFIED (MERGE + CHOICE)
    return cvdf
