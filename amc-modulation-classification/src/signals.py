"""Signal generation and preprocessing for Automatic Modulation Classification.

Each example is a burst of 128 complex baseband samples stored as a
(2, 128) float array: row 0 = in-phase (I), row 1 = quadrature (Q).
This matches the layout of the RadioML 2016.10A dataset.
"""

import numpy as np

MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
N_SAMPLES = 128
SPS = 8  # samples per symbol for digital modulations


# ---------------------------------------------------------------- helpers
def _qam_points(m):
    k = int(np.sqrt(m))
    levels = np.arange(-(k - 1), k, 2)
    pts = np.array([complex(i, q) for i in levels for q in levels])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _psk_points(m):
    return np.exp(1j * (2 * np.pi * np.arange(m) / m + (np.pi / m if m > 2 else 0)))


def _upsample(symbols, sps, rng):
    """Rectangular pulse shaping followed by light smoothing."""
    x = np.repeat(symbols, sps)
    taps = np.hanning(sps + 1)
    taps /= taps.sum()
    return np.convolve(x, taps, mode="same")


def _message(n, rng):
    """Band-limited random 'audio' message for analog modulations."""
    t = np.arange(n)
    msg = np.zeros(n)
    for _ in range(3):
        f = rng.uniform(0.005, 0.03)
        msg += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return msg / np.max(np.abs(msg))


def _fsk(bits, sps, h, gaussian, rng):
    freq = np.repeat(2 * bits - 1.0, sps)
    if gaussian:
        g = np.exp(-0.5 * (np.arange(-sps, sps + 1) / (0.5 * sps)) ** 2)
        freq = np.convolve(freq, g / g.sum(), mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps
    return np.exp(1j * phase)


# ---------------------------------------------------------------- modulator
def modulate(mod, n=N_SAMPLES, rng=None):
    """Return n clean complex baseband samples for the given modulation."""
    rng = rng or np.random.default_rng()
    n_sym = n // SPS + 4
    if mod == "BPSK":
        s = _upsample(_psk_points(2)[rng.integers(0, 2, n_sym)], SPS, rng)
    elif mod == "QPSK":
        s = _upsample(_psk_points(4)[rng.integers(0, 4, n_sym)], SPS, rng)
    elif mod == "8PSK":
        s = _upsample(_psk_points(8)[rng.integers(0, 8, n_sym)], SPS, rng)
    elif mod == "QAM16":
        s = _upsample(_qam_points(16)[rng.integers(0, 16, n_sym)], SPS, rng)
    elif mod == "QAM64":
        s = _upsample(_qam_points(64)[rng.integers(0, 64, n_sym)], SPS, rng)
    elif mod == "PAM4":
        lv = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        s = _upsample(lv[rng.integers(0, 4, n_sym)].astype(complex), SPS, rng)
    elif mod == "CPFSK":
        s = _fsk(rng.integers(0, 2, n_sym), SPS, 0.5, False, rng)
    elif mod == "GFSK":
        s = _fsk(rng.integers(0, 2, n_sym), SPS, 0.5, True, rng)
    elif mod == "AM-DSB":
        m = _message(n_sym * SPS, rng)
        s = (1 + 0.5 * m).astype(complex)
    elif mod == "AM-SSB":
        m = _message(n_sym * SPS, rng)
        spec = np.fft.fft(m)
        spec[len(spec) // 2:] = 0  # keep one sideband (analytic signal)
        s = np.fft.ifft(2 * spec)
    elif mod == "WBFM":
        m = _message(n_sym * SPS, rng)
        s = np.exp(1j * 2 * np.pi * 0.08 * np.cumsum(m))
    else:
        raise ValueError(f"Unknown modulation: {mod}")
    start = rng.integers(0, len(s) - n)
    return s[start:start + n]


def channel(x, snr_db, rng=None):
    """Apply a simple wireless channel: random phase, small frequency
    offset, flat Rayleigh-like gain and AWGN at the given SNR."""
    rng = rng or np.random.default_rng()
    n = np.arange(len(x))
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    phase = rng.uniform(0, 2 * np.pi)
    f_off = rng.uniform(-0.002, 0.002)
    gain = np.abs(rng.normal(1, 0.15))
    y = gain * x * np.exp(1j * (phase + 2 * np.pi * f_off * n))
    noise_power = gain ** 2 / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    return y + noise


def to_iq(x):
    """Complex vector -> (2, N) float32 array."""
    return np.stack([x.real, x.imag]).astype(np.float32)


def normalize(iq):
    """Scale a (2, N) or (B, 2, N) I/Q array to unit average power."""
    iq = np.asarray(iq, dtype=np.float32)
    p = np.sqrt(np.mean(iq[..., 0, :] ** 2 + iq[..., 1, :] ** 2, axis=-1, keepdims=True))
    return iq / (p[..., None] + 1e-8)


def generate_dataset(per_class_per_snr=200, snrs=range(-20, 20, 2), seed=0):
    """Synthetic RadioML-style dataset.

    Returns X (N, 2, 128), y (N,) integer labels, snr (N,).
    """
    rng = np.random.default_rng(seed)
    X, y, s = [], [], []
    for label, mod in enumerate(MODULATIONS):
        for snr in snrs:
            for _ in range(per_class_per_snr):
                X.append(to_iq(channel(modulate(mod, rng=rng), snr, rng)))
                y.append(label)
                s.append(snr)
    return np.array(X), np.array(y), np.array(s)


def load_radioml(path):
    """Load the RadioML 2016.10A pickle (RML2016.10a_dict.pkl).

    Returns X (N, 2, 128), y (N,), snr (N,) with labels ordered as MODULATIONS.
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    X, y, s = [], [], []
    for (mod, snr), arr in data.items():
        X.append(arr)
        y += [MODULATIONS.index(mod)] * len(arr)
        s += [snr] * len(arr)
    return np.concatenate(X).astype(np.float32), np.array(y), np.array(s)


def load_signal_file(file_or_path, name=""):
    """Read an uploaded .npy or .csv signal and return a (2, 128) array.

    Accepted layouts: (2, N), (N, 2) or a complex vector of length N.
    Longer signals are cut to the first 128 samples.
    """
    name = (name or str(file_or_path)).lower()
    if name.endswith(".npy"):
        arr = np.load(file_or_path)
    elif name.endswith(".csv") or name.endswith(".txt"):
        arr = np.loadtxt(file_or_path, delimiter=",")
    else:
        raise ValueError("Please upload a .npy or .csv file")
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = to_iq(arr.ravel())
    elif arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] == 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] < N_SAMPLES:
        raise ValueError(f"Expected 2 x {N_SAMPLES} I/Q samples, got shape {arr.shape}")
    return arr[:, :N_SAMPLES].astype(np.float32)
