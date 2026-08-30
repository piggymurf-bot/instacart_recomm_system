import numpy as np

class FaronF1Optimizer:
    """
    Exact Expected F1 Maximization for basket prediction (Faron's Algorithm).
    Optimizes candidate selection per user given predicted probabilities.
    """
    @staticmethod
    def _compute_poisson_binomial_dp(probs: np.ndarray) -> np.ndarray:
        """Computes DP table for distribution of true basket size s."""
        n = len(probs)
        dp = np.zeros((n + 1, n + 1), dtype=np.float64)
        dp[0, 0] = 1.0

        for i in range(1, n + 1):
            p = probs[i - 1]
            dp[i, 0] = dp[i - 1, 0] * (1.0 - p)
            for j in range(1, i + 1):
                dp[i, j] = dp[i - 1, j] * (1.0 - p) + dp[i - 1, j - 1] * p

        return dp[n, :]  # P(s) array of size n + 1

    @classmethod
    def get_optimal_k(cls, probs: np.ndarray) -> int:
        """
        Given sorted probabilities (descending) for one user,
        finds the integer k (number of top items to pick) maximizing Expected F1.
        """
        n = len(probs)
        if n == 0:
            return 0

        # Compute P(s), distribution over actual basket sizes
        P_s = cls._compute_poisson_binomial_dp(probs)

        best_k = 0
        best_expected_f1 = P_s[0]  # k=0 (predicting 'None'), F1 is 1 if s=0, else 0

        # Precompute cumulative sum of probabilities for fast expectation checks
        cumsum_p = np.cumsum(probs)

        # Iterate over possible basket sizes k (1 to n)
        for k in range(1, n + 1):
            expected_c_total = cumsum_p[k - 1]
            sum_p_all = cumsum_p[-1]

            expected_f1_k = 0.0
            for s in range(1, n + 1):
                if P_s[s] > 1e-9:
                    # Expected overlap c for top-k items given true size s
                    c_s = expected_c_total * (s / sum_p_all) if sum_p_all > 0 else 0
                    c_s = min(c_s, float(min(k, s)))
                    expected_f1_k += (2.0 * c_s / (k + s)) * P_s[s]

            if expected_f1_k > best_expected_f1:
                best_expected_f1 = expected_f1_k
                best_k = k

        return best_k