import numpy as np 

def kl_divergence(p, q):
    """Compute the Kullback-Leibler divergence between two distributions.

    The KL divergence is defined as
    :math:`D_{KL}(p, q) = \sum_x p(x_i) * (\log p(x_i) - \log q(x_i))`
    which can be rewritten as
    :math:`D_{KL}(p, q) = \sum_x p(x_i) * \log \frac{p(x_i)}{q(x_i)}`
    and is computationally more conventient.
    Some interesting properties of the KL divergence:
      - The KL divergence is always non-negative, i.e.
        :math:`D_{KL}(p, q) \geq 0`.
      - The KL divergence is additive for independent distributions, i.e.
        :math:`D_{KL}(P, Q) = D_{KL}(P_1, Q_1) + D_{KL}(P_2, Q_2)`.
    """
    # Ensure tensors are numpy arrays
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize the distributions to sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Add a small constant to avoid division by zero or log of zero
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    # Compute KL divergence
    kl_div = np.sum(p * np.log(p / q))
    return kl_div


   