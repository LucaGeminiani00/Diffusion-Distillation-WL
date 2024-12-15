import numpy as np
import scipy
from Models.ts2vec.ts2vec import TS2Vec


class ContextFIDCalculator():
    """
    Class to calculate the Context-FID score for time-series data.
    Original and generated data are mandatory inputs at initialization.
    """

    def __init__(self, ori_data, device=0, batch_size=8, lr=0.001, output_dims=320, max_train_length=3000):
        self.ori_data = ori_data

        self.model = TS2Vec(
            input_dims=ori_data.shape[-1],
            device=device,
            batch_size=batch_size,
            lr=lr,
            output_dims=output_dims,
            max_train_length=max_train_length
        )
        self.is_trained = False

    def train(self, verbose=False):
        self.model.fit(self.ori_data, verbose=verbose)
        self.is_trained = True

    def compute_fid(self, generated_data):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before computing FID.")

        ori_representation = self.model.encode(self.ori_data, encoding_window='full_series')
        gen_representation = self.model.encode(generated_data, encoding_window='full_series')

        idx = np.random.permutation(self.ori_data.shape[0])
        ori_representation = ori_representation[idx]
        gen_representation = gen_representation[idx]

        fid_score = self._calculate_fid(ori_representation, gen_representation)
        return fid_score

    @staticmethod
    def _calculate_fid(act1, act2):
        """
        Static method to calculate FID given two activation matrices.
        """
        # Calculate mean and covariance
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Square root of the product of covariances
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
