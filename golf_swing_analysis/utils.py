# utils.py
import numpy as np
import math

def angle_3pt(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cosv = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def line_angle(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return float(np.degrees(math.atan2(dy, dx)))

def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    if x.ndim == 1:
        pad = window//2
        x_p = np.pad(x, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(x_p, kernel, mode='valid')
    else:
        out = np.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i] = smooth_series(x[:, i], window)
        return out

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
