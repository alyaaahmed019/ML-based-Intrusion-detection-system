"""
Decision-rule helpers shared between training (threshold/bias search during
evaluation) and inference (applying the chosen threshold/bias at predict time).
"""

import numpy as np


def predict_with_proba_threshold(y_prob: np.ndarray, threshold_normal: float = 0.5) -> np.ndarray:
    """
    Classify a window as Normal (class 0) only if P(Normal) >= threshold;
    otherwise pick the highest-probability attack class. This lets us trade
    off false alarms vs. missed attacks by tuning threshold_normal.
    """
    preds = np.empty(y_prob.shape[0], dtype=int)
    is_normal = y_prob[:, 0] >= threshold_normal
    preds[is_normal] = 0
    if not np.all(is_normal):
        preds[~is_normal] = np.argmax(y_prob[~is_normal, 1:], axis=1) + 1
    return preds


def predict_with_decision_bias(y_score: np.ndarray, bias_normal: float = 0.0) -> np.ndarray:
    """Same idea as predict_with_proba_threshold but for decision_function
    scores (e.g. LinearSVC), which have no probabilities to threshold."""
    scores = y_score.copy()
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores[:, 0] += bias_normal
    return np.argmax(scores, axis=1)
