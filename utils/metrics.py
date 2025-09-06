import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def DMSE(pred, true, ref=None, strength: float = 2.0, eps: float = 1e-8):
    """
    Deviation-weighted MSE (DMSE): emphasize errors occurring during large volatility.
    - ref: reference series to measure volatility; if None, uses true.
    - strength: scaling factor for volatility weights (>=0). Higher -> more emphasis.

    Implementation:
    1) Compute volatility v = |Δ ref| along time for each channel, pad to length T by repeating last diff.
    2) Normalize v to [0,1] per-batch per-channel: v_norm = (v - v.min) / (v.max - v.min + eps)
    3) Weight w = 1 + strength * v_norm
    4) DMSE = mean(w * (pred-true)^2)
    """
    if ref is None:
        ref = true
    # Ensure numpy arrays
    pred = np.asarray(pred)
    true = np.asarray(true)
    ref = np.asarray(ref)

    # Shapes: [B, T, C]
    assert pred.shape == true.shape, "pred and true must have the same shape [B, T, C]"
    B, T, C = ref.shape
    if T < 2:
        return MSE(pred, true)

    # Volatility along time
    diff = np.abs(np.diff(ref, axis=1))  # [B, T-1, C]
    # Pad to T by repeating last diff
    v = np.concatenate([diff, diff[:, -1:, :]], axis=1)  # [B, T, C]

    # Normalize per sample-channel
    v_min = v.min(axis=1, keepdims=True)
    v_max = v.max(axis=1, keepdims=True)
    v_norm = (v - v_min) / (v_max - v_min + eps)

    w = 1.0 + strength * v_norm  # [B, T, C]
    dmse = np.mean(w * (pred - true) ** 2)
    return dmse
