"""Joint question-cluster resampling with the five shared U endpoints kept dependent."""
import numpy as np


def paired_difference_matrix(filtered, uniform):
    filtered, uniform = np.asarray(filtered, dtype=float), np.asarray(uniform, dtype=float)
    if filtered.ndim != 3 or uniform.ndim != 2 or filtered.shape[1:] != uniform.shape:
        raise ValueError("Require F[selection,training,question] and U[training,question]")
    return (filtered-uniform[None, :, :]).mean(axis=2)*100


def joint_cluster_accuracies(correct, cluster_ids, replicates=20000, seed=20261116):
    """One identical cluster draw per row for EVERY fixed model, including anchors."""
    correct = np.asarray(correct, dtype=float)
    if correct.ndim != 2 or correct.shape[1] != len(cluster_ids) or correct.shape[1] == 0:
        raise ValueError("Correctness must be model-by-question on one complete question set")
    if not np.isin(correct, [0, 1]).all() or replicates < 10000:
        raise ValueError("Need binary fixed predictions and at least10000 resamples")
    labels, inverse = np.unique(cluster_ids, return_inverse=True)
    sizes = np.bincount(inverse).astype(float)
    sums = np.stack([np.bincount(inverse, weights=row, minlength=len(labels)) for row in correct])
    rng = np.random.default_rng(seed)
    outputs = []
    for offset in range(0, replicates, 500):
        n = min(500, replicates-offset)
        weights = rng.multinomial(len(labels), np.full(len(labels), 1/len(labels)), size=n)
        outputs.append((weights @ sums.T)/(weights @ sizes)[:, None])
    return np.concatenate(outputs)


def intervals_from_joint(joint, keys, training_seeds, probe_seeds, *, crossed=False, seed=20261118):
    """Crossed mode samples training blocks and paired F/R selection blocks, never15 U copies."""
    indices = {key: i for i, key in enumerate(keys)}
    U = np.array([indices[f"U-s{s}"] for s in training_seeds])
    F = np.array([[indices[f"F-j{j}-s{s}"] for s in training_seeds] for j in probe_seeds])
    R = np.array([[indices[f"R-j{j}-s{s}"] for s in training_seeds] for j in probe_seeds])
    n = joint.shape[0]
    if crossed:
        rng = np.random.default_rng(seed)
        ss = rng.integers(0, len(training_seeds), (n, len(training_seeds)))
        jj = rng.integers(0, len(probe_seeds), (n, len(probe_seeds)))
        row = np.arange(n)[:, None, None]
        f = joint[row, F[jj[:, :, None], ss[:, None, :]]].mean(axis=(1, 2))
        r = joint[row, R[jj[:, :, None], ss[:, None, :]]].mean(axis=(1, 2))
        u = joint[np.arange(n)[:, None], U[ss]].mean(axis=1)
    else:
        f = joint[:, F].mean(axis=(1, 2))
        r = joint[:, R].mean(axis=(1, 2))
        u = joint[:, U].mean(axis=1)
    differences = {"F-U": (f-u)*100, "F-R": (f-r)*100,
        "F-S2": (f-joint[:, indices["S2"]])*100, "F-A": (f-joint[:, indices["A"]])*100}
    return {key: {"percentile95_pp": np.quantile(value, [0.025, 0.975]).tolist(),
        "replicates": n, "scope": "crossed training/selection/question-cluster exploratory" if crossed else "question-cluster conditional on fixed training/probe seeds"}
        for key, value in differences.items()}
