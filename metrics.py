import torch
import numpy as np
from scipy.linalg import sqrtm

def _to_numpy_trimmed(tensor):
    """
    converte in numpy e toglie padding
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor
    arr_3d = arr.reshape(-1, 50, 3)

    all_zero = np.all(arr_3d == 0, axis=(1, 2))
    zero_indices = np.where(all_zero)[0]
    if zero_indices.size > 0:
        arr_3d = arr_3d[:zero_indices[0]]  # taglia al padding

    return arr_3d

def mpjpe(references, hypotheses):
    assert len(references) == len(hypotheses)
    mpjpe_value = 0.0

    for pred, true in zip(hypotheses, references):
        pred_3d = _to_numpy_trimmed(pred)
        true_3d = _to_numpy_trimmed(true)
        min_len = min(len(pred_3d), len(true_3d))
        if min_len == 0:
            continue
        pred_3d = pred_3d[:min_len]
        true_3d = true_3d[:min_len]
        mpjpe_value += _p_mpjpe(pred_3d, true_3d)

    return float(mpjpe_value)


def _p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target,    axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True)) + 1e-8
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True)) + 1e-8
    X0 /= normX
    Y0 /= normY

    H  = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V  = Vt.transpose(0, 2, 1)
    R  = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a= tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a* np.matmul(predicted, R) + t
    return float(np.mean(
        np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1)
    ))


def _angle_between(v1, v2):
    ref = np.array([1.0, 0.0, 0.0])
    cos1 = np.dot(v1, ref) / (np.linalg.norm(v1) * np.linalg.norm(ref) + 1e-8)
    cos2 = np.dot(v2, ref) / (np.linalg.norm(v2) * np.linalg.norm(ref) + 1e-8)
    a1 = np.degrees(np.arccos(np.clip(cos1, -1., 1.)))
    a2 = np.degrees(np.arccos(np.clip(cos2, -1., 1.)))
    return abs(a1 - a2)


def mpjae(references, hypotheses):
    assert len(references) == len(hypotheses)
    skeleton = getSkeletalModelStructure()
    scores   = []

    for pred, true in zip(hypotheses, references):
        pred_3d = _to_numpy_trimmed(pred)
        true_3d = _to_numpy_trimmed(true)

        min_len = min(len(pred_3d), len(true_3d))
        if min_len == 0:
            continue
        pred_3d = pred_3d[:min_len]
        true_3d = true_3d[:min_len]

        frame_errors = []
        for t in range(min_len):
            bone_errors = []
            for parent_idx, child_idx in skeleton:
                pv = pred_3d[t, child_idx] - pred_3d[t, parent_idx]
                gv = true_3d[t, child_idx] - true_3d[t, parent_idx]
                bone_errors.append(_angle_between(pv, gv))
            frame_errors.append(np.mean(bone_errors))

        if frame_errors:
            scores.append(np.mean(frame_errors))

    return float(np.mean(scores)) if scores else float('nan')

def _extract_features(seq_2d):
    """
    estrae media, velocità, accelerazione, dev std
    """
    T, D = seq_2d.shape
    if T == 0:
        return None
        
    #media
    mean_pose = np.mean(seq_2d, axis=0)
    # dev std
    std_pose = np.std(seq_2d, axis=0)
    
    # velocità
    if T > 1:
        vel = seq_2d[1:] - seq_2d[:-1]
        mean_vel = np.mean(vel, axis=0)
    else:
        mean_vel = np.zeros(D)
        
    #accellerazione
    if T > 2:
        acc = vel[1:] - vel[:-1]
        mean_acc = np.mean(acc, axis=0)
    else:
        mean_acc = np.zeros(D)
        
    #concatena 
    return np.concatenate([mean_pose, mean_vel, mean_acc, std_pose])

def fid(references, hypotheses):
    """
    fid diversa
    """
    assert len(references) == len(hypotheses)
    true_features = []
    pred_features = []

    for pred, true in zip(hypotheses, references):
        pred_3d = _to_numpy_trimmed(pred) #pred
        true_3d = _to_numpy_trimmed(true) #gt
        
        pred_2d = pred_3d.reshape(pred_3d.shape[0], -1)
        true_2d = true_3d.reshape(true_3d.shape[0], -1)
        
        t_feat = _extract_features(true_2d)
        p_feat = _extract_features(pred_2d)
        
        if t_feat is not None and p_feat is not None:
            true_features.append(t_feat)
            pred_features.append(p_feat)

    if len(true_features) == 0:
        return float('nan')

    true_matrix = np.array(true_features)
    pred_matrix = np.array(pred_features)

    mu1 = np.mean(true_matrix, axis=0)
    sigma1 = np.cov(true_matrix, rowvar=False)
    mu2 = np.mean(pred_matrix, axis=0)
    sigma2 = np.cov(pred_matrix, rowvar=False)

    #eps perchè matrice grande
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0])*eps
    sigma2 += np.eye(sigma2.shape[0])*eps

    diff = mu1 - mu2
    mean_diff = np.dot(diff, diff)

    covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = mean_diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return float(fid_score)


def getSkeletalModelStructure():
    return (
        # head
        (1, 0),

        (1, 2),

        # left arm
        (2, 3),

        (3, 4),      # 舍弃

        (1, 5),

        (5, 6),

        (6, 7),     # 舍弃

        (7, 8),

        (8, 9),

        (9, 10),

        (10, 11),

        (11, 12),

        (8, 13),

        (13, 14),

        (14, 15),

        (15, 16),

        (8, 17),

        (17, 18),

        (18, 19),

        (19, 20),

        (8, 21),

        (21, 22),

        (22, 23),

        (23, 24),

        (8, 25),

        (25, 26),

        (26, 27),

        (27, 28),

        (4, 29),

        (29, 30),

        (30, 31),

        (31, 32),

        (32, 33),

        (29, 34),

        (34, 35),

        (35, 36),

        (36, 37),

        (29, 38),

        (38, 39),

        (39, 40),

        (40, 41),

        (29, 42),

        (42, 43),

        (43, 44),

        (44, 45),

        (29, 46),

        (46, 47),

        (47, 48),

        (48, 49),

    )
