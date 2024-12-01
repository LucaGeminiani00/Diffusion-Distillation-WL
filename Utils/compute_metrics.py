import numpy as np
import torch

from Utils.context_fid import Context_FID
from Utils.correlational_score import *
from Utils.discriminative_score import discriminative_score
from Utils.predictive_score import predictive_score


def pearson_corr(actual, predic):
    if actual.shape[0] > predic.shape[0]:
        actual = actual[:predic.shape[0]]
    elif predic.shape[0] > actual.shape[0]:
        predic = predic[:actual.shape[0]]

    a_mean = np.mean(actual, axis=1, keepdims=True)
    p_mean = np.mean(predic, axis=1, keepdims=True)

    a_diff = actual - a_mean
    p_diff = predic - p_mean

    numerator = np.sum(a_diff * p_diff, axis=1)
    denominator = np.sqrt(np.sum(a_diff ** 2, axis=1)) * np.sqrt(np.sum(p_diff ** 2, axis=1))
    correlation = numerator / (denominator + 1e-8)

    return abs(np.mean(correlation))


def compute_metrics(ori_data, fake_data, train=False):
    """
    Compute metrics for original and fake data.

    Args:
        ori_data (numpy.ndarray): Original data.
        fake_data (numpy.ndarray): Fake/generated data.
        train (bool): If True, compute all metrics including those requiring training

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    FID_score = Context_FID(ori_data, fake_data)
    pearson_correlation = pearson_corr(ori_data, fake_data)
    
    ori = torch.from_numpy(ori_data)
    fake = torch.from_numpy(fake_data)
    corr = CrossCorrelLoss(ori)
    correl_score = corr.compute(fake)
    
    metrics = {
        "FID_score": FID_score,
        "correlational_score": correl_score,
        "Pearson's correlation": pearson_correlation
    }
    
    if train:
        discriminative = discriminative_score(ori_data, fake_data)
        predictive_score_value = predictive_score(ori_data, fake_data)
        metrics.update({
            "discriminative_score": discriminative,
            "predictive_score": predictive_score_value
        })
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics
