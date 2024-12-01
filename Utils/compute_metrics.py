import bz2

import numpy as np

from Utils.context_fid import Context_FID
from Utils.discriminative_score import discriminative_score
from Utils.predictive_score import predictive_score


def compute_metrics(ori_data, fake_data): 
    FID_score = Context_FID(ori_data, fake_data)
    discriminative, fake_accuracy, real_accuracy = discriminative_score(ori_data, fake_data)
    predictive_score_value = predictive_score(ori_data, fake_data)
    
    metrics = {
        "FID_score": FID_score,
        "discriminative_score": discriminative,
        "fake_data_accuracy": fake_accuracy,
        "real_data_accuracy": real_accuracy,
        "predictive_score": predictive_score_value
    }
    for key, value in metrics.items():
      print(f"{key}: {value}")
    
    return metrics



def calc_correlation(actual, predic):
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