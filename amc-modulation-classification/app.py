"""SignalSense - Streamlit web app for Automatic Modulation Classification.

Run:  streamlit run app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.signals import MODULATIONS, channel, load_signal_file, modulate, normalize, to_iq

MODEL_PATH = os.path.join("models", "amc_cnn.keras")

st.set_page_config(page_title="SignalSense - AMC", page_icon="📡", layout="wide")


@st.cache_resource
def load_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


def predict(model, iq):
    x = normalize(iq).T[None, ...]  # (1, 128, 2)
    return model.predict(x, verbose=0)[0]


def plot_constellation(iq):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(iq[0], iq[1], s=10, alpha=0.7)
    lim = 1.1 * np.max(np.abs(iq))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("In-phase (I)"); ax.set_ylabel("Quadrature (Q)")
    ax.set_title("Constellation diagram"); ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_time(iq):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(iq[0], label="I"); ax.plot(iq[1], label="Q")
    ax.set_xlabel("Sample"); ax.set_title("I/Q waveform"); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page
st.caption("WIRELESS MOBILE COMMUNICATION · MODULATION CLASSIFICATION")
st.title("Know the signal before you demodulate.")
st.write(
    "Upload a received I/Q signal to see its constellation diagram and the modulation "
    "type predicted by the 1D CNN model. This tool is intended for learning and "
    "demonstration, not for operational receivers."
)

left, right = st.columns([1.45, 1])
with right:
    st.subheader("One signal, eleven classes")
    st.write("The model returns a probability for each modulation:")
    st.write(" · ".join(MODULATIONS))
    st.caption("Signals are normalised to unit power before inference.")

with left:
    st.markdown("**01 / SIGNAL INPUT**")
    uploaded = st.file_uploader("Choose an I/Q signal file (2 × 128 samples)", type=["npy", "csv"])
    with st.expander("No file? Generate a test signal"):
        c1, c2 = st.columns(2)
        test_mod = c1.selectbox("Modulation", MODULATIONS, index=MODULATIONS.index("QPSK"))
        test_snr = c2.slider("SNR (dB)", -10, 18, 10, step=2)
        use_test = st.button("Generate signal")

iq = None
if uploaded is not None:
    try:
        iq = load_signal_file(uploaded, uploaded.name)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
elif use_test:
    iq = to_iq(channel(modulate(test_mod), test_snr))
    st.session_state["test_iq"] = iq
elif "test_iq" in st.session_state:
    iq = st.session_state["test_iq"]

if iq is not None:
    st.divider()
    a, b = st.columns([1, 1.3])
    a.pyplot(plot_constellation(iq))
    b.pyplot(plot_time(iq))

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Run `python train.py` first to create models/amc_cnn.keras.")
    else:
        probs = predict(load_model(), iq)
        top = int(np.argmax(probs))
        st.success(f"Predicted modulation: **{MODULATIONS[top]}**  ·  confidence {probs[top]:.1%}")
        order = np.argsort(probs)[::-1][:5]
        st.bar_chart({"probability": {MODULATIONS[i]: float(probs[i]) for i in order}})

st.divider()
st.caption("SignalSense is a demonstration project. It is not a certified communication "
           "system; results at very low SNR may be unreliable.")
