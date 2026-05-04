import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

EPS = 1e-8

def entropy_np(p):
    return -(p * np.log2(p + EPS)).sum(axis=1)

def kl_divergence_np(p, q):
    return (p * (np.log((p + EPS) / (q + EPS)))).sum(axis=1)

def js_divergence_np(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_np(p, m) + 0.5 * kl_divergence_np(q, m)

def cosine_np(p, q):
    return np.array([
        cosine_similarity(p[i].reshape(1, -1), q[i].reshape(1, -1))[0, 0]
        for i in range(len(p))
    ])

def precision_at_k(true_entropy, pred_entropy, k):
    true_top = set(np.argsort(-true_entropy)[:k])
    pred_top = set(np.argsort(-pred_entropy)[:k])
    return len(true_top.intersection(pred_top)) / k

def evaluate_distribution_metrics(true_probs, pred_probs):
    true_entropy = entropy_np(true_probs)
    pred_entropy = entropy_np(pred_probs)

    kl = kl_divergence_np(true_probs, pred_probs)
    jsd = js_divergence_np(true_probs, pred_probs)
    cos = cosine_np(true_probs, pred_probs)

    pearson = pearsonr(true_entropy, pred_entropy)[0]
    spearman = spearmanr(true_entropy, pred_entropy)[0]

    results = {
        "KL_mean": float(np.mean(kl)),
        "KL_std": float(np.std(kl)),
        "JSD_mean": float(np.mean(jsd)),
        "JSD_std": float(np.std(jsd)),
        "Cosine_mean": float(np.mean(cos)),
        "Cosine_std": float(np.std(cos)),
        "Pearson_entropy": float(pearson),
        "Spearman_entropy": float(spearman),
        "Precision@100": float(precision_at_k(true_entropy, pred_entropy, 100)),
        "Precision@200": float(precision_at_k(true_entropy, pred_entropy, 200)),
        "Precision@500": float(precision_at_k(true_entropy, pred_entropy, 500)),
    }

    return results, true_entropy, pred_entropy
