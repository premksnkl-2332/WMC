"""Write example I/Q signal files to samples/ for testing the web app."""

import os

import numpy as np

from src.signals import MODULATIONS, channel, modulate, to_iq

os.makedirs("samples", exist_ok=True)
rng = np.random.default_rng(7)
for mod in MODULATIONS:
    iq = to_iq(channel(modulate(mod, rng=rng), snr_db=14, rng=rng))
    np.save(f"samples/{mod}_14dB.npy", iq)
np.savetxt("samples/QPSK_14dB.csv", np.load("samples/QPSK_14dB.npy"), delimiter=",")
print("Wrote", len(MODULATIONS) + 1, "files to samples/")
