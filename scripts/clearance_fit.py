"""
Clearance model (library; used by 12_clearance_cohort_clearance_fit and independent_scan (13, 14)):
detections summed per 5-min bin; single exponential f(t) = exp(-t / tau) (N0 fixed at 1,
plateau 0) and double exponential N0 exp(-t / tau1) + N1 exp(-t / tau2), compared by
reduced chi-squared.
"""
import numpy as np

BIN_H = 5 / 60


def single(t, tau):
    return np.exp(-t / tau)


def double(t, n0, tau1, n1, tau2):
    return n0 * np.exp(-t / tau1) + n1 * np.exp(-t / tau2)


def red_chi2(y, yhat, k):
    return float(np.sum((y - yhat) ** 2) / max(len(y) - k, 1))


def binned(t, hours):
    """Counts of detection times t (h) per complete BIN_H bin; time = bin start (t = 0 for the first bin)."""
    edges = np.arange(0, hours + 1e-9, BIN_H)
    edges = edges[edges + BIN_H <= hours + 1e-9]             # complete bins only
    return edges, np.histogram(t, bins=np.append(edges, edges[-1] + BIN_H))[0]
