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