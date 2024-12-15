import numpy as np
import torch

from Utils.context_fid import ContextFIDCalculator
from Utils.correlational_score import *
from Utils.discriminative_score import DiscriminativeScoreModel
from Utils.predictive_score import PredictiveScoreModel


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


def compute_metrics(ori_data, fake_data, predictors=None):
    """
    Compute metrics for original and fake data.
    """
  
    pearson_correlation = pearson_corr(ori_data, fake_data)
    
    ori = torch.from_numpy(ori_data)
    fake = torch.from_numpy(fake_data)
    corr = CrossCorrelLoss(ori)
    correl_score = corr.compute(fake)
    
    metrics = {
        "Correlational score": correl_score,
        "Pearson's correlation": pearson_correlation
    }
    
    if predictors is not None:
        fid, dis, pred = predictors[:3] 
        FID_score = fid.compute_fid(fake_data)
        discriminative = dis.compute_dis(fake_data)
        predictive_score_value = pred.compute_pred(fake_data)
        metrics.update({
            "FID score": FID_score,
            "Discriminative score": discriminative,
            "Predictive score": predictive_score_value
        })

    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

def train_metrics(ori_data,fake_data): 
    fid = ContextFIDCalculator(ori_data)
    fid.train() 

    dis = DiscriminativeScoreModel(ori_data, fake_data)
    dis.train()

    pred = PredictiveScoreModel(ori_data, fake_data)
    pred.train()

    return [fid, dis, pred]

