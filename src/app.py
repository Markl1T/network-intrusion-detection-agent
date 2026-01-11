import streamlit as st
import pandas as pd
import joblib
import random
import ipaddress

from config import FEATURES, MODELS_DIR, SAMPLED_PATH
from preprocessing import clean_features


st.set_page_config(
    page_title="Network Intrusion Detection Agent",
    layout="wide",
)

st.title("Network Intrusion Detection Agent")


@st.cache_resource
def load_models():
    anomaly = joblib.load(MODELS_DIR / "anomaly_iforest.pkl")
    stage1 = joblib.load(MODELS_DIR / "stage1_xgb.pkl")
    stage2, encoder = joblib.load(MODELS_DIR / "stage2_xgb.pkl")
    return anomaly, stage1, stage2, encoder

anomaly_model, stage1_model, stage2_model, encoder = load_models()


@st.cache_data
def load_sample_pool():
    return pd.read_csv(SAMPLED_PATH)

sample_pool = load_sample_pool()

def random_ip():
    return str(ipaddress.IPv4Address(random.randint(0x0B000000, 0xDF000000)))


if "base_row" not in st.session_state:
    st.session_state.base_row = sample_pool.sample(1).iloc[0]

if st.button("Load Random Real Flow"):
    st.session_state.base_row = sample_pool.sample(1).iloc[0]
    
    st.session_state.src_ip = st.session_state.base_row["IPV4_SRC_ADDR"]
    st.session_state.dst_ip = st.session_state.base_row["IPV4_DST_ADDR"]
    st.session_state.src_port = int(st.session_state.base_row["L4_SRC_PORT"])
    st.session_state.dst_port = int(st.session_state.base_row["L4_DST_PORT"])
    st.session_state.protocol = int(st.session_state.base_row["PROTOCOL"])
    st.session_state.tcp_flags = int(st.session_state.base_row["TCP_FLAGS"])
    st.session_state.l7_proto = float(st.session_state.base_row["L7_PROTO"])
    st.session_state.in_bytes = int(st.session_state.base_row["IN_BYTES"])
    st.session_state.out_bytes = int(st.session_state.base_row["OUT_BYTES"])
    st.session_state.in_pkts = int(st.session_state.base_row["IN_PKTS"])
    st.session_state.out_pkts = int(st.session_state.base_row["OUT_PKTS"])
    st.session_state.flow_duration = int(st.session_state.base_row["FLOW_DURATION_MILLISECONDS"])


row = st.session_state.base_row

c1, c2, c3, c4 = st.columns(4)

with c1:
    src_ip = st.text_input("IPv4 Src Addr", value=st.session_state.get("src_ip", row["IPV4_SRC_ADDR"]))
    src_port = st.number_input("L4 Src Port", 0, 65535, value=int(st.session_state.get("src_port", row["L4_SRC_PORT"])))
    in_bytes = st.number_input("IN_BYTES", 0, 1_000_000, value=int(st.session_state.get("in_bytes", row["IN_BYTES"])))

with c2:
    dst_ip = st.text_input("IPv4 Dst Addr", value=st.session_state.get("dst_ip", row["IPV4_DST_ADDR"]))
    dst_port = st.number_input("L4 Dst Port", 0, 65535, value=int(st.session_state.get("dst_port", row["L4_DST_PORT"])))
    out_bytes = st.number_input("OUT_BYTES", 0, 1_000_000, value=int(st.session_state.get("out_bytes", row["OUT_BYTES"])))

with c3:
    protocol = st.number_input("PROTOCOL", 0, 255, value=int(st.session_state.get("protocol", row["PROTOCOL"])))
    tcp_flags = st.number_input("TCP_FLAGS", 0, 255, value=int(st.session_state.get("tcp_flags", row["TCP_FLAGS"])))
    in_pkts = st.number_input("IN_PKTS", 0, 10_000, value=int(st.session_state.get("in_pkts", row["IN_PKTS"])))

with c4:
    l7_proto = st.number_input("L7_PROTO", 0.0, 100.0, value=float(st.session_state.get("l7_proto", row["L7_PROTO"])))
    flow_duration = st.number_input("FLOW_DURATION_MS", 0, 60_000, value=int(st.session_state.get("flow_duration", row["FLOW_DURATION_MILLISECONDS"])))
    out_pkts = st.number_input("OUT_PKTS", 0, 10_000, value=int(st.session_state.get("out_pkts", row["OUT_PKTS"])))


if st.button("Run Detection", type="primary"):
    full_row = st.session_state.base_row.copy()

    full_row.update({
        "IPV4_SRC_ADDR": src_ip,
        "IPV4_DST_ADDR": dst_ip,
        "L4_SRC_PORT": src_port,
        "L4_DST_PORT": dst_port,
        "PROTOCOL": protocol,
        "TCP_FLAGS": tcp_flags,
        "L7_PROTO": l7_proto,
        "IN_BYTES": in_bytes,
        "OUT_BYTES": out_bytes,
        "IN_PKTS": in_pkts,
        "OUT_PKTS": out_pkts,
        "FLOW_DURATION_MILLISECONDS": flow_duration
    })

    raw_df = pd.DataFrame([full_row])
    X = clean_features(raw_df, FEATURES)

    st.markdown("---")

    # ---- STAGE 0: ANOMALY ----
    anomaly_score = -anomaly_model.decision_function(X)[0]
    st.metric("Anomaly Score", f"{anomaly_score:.4f}")

    # ---- STAGE 1: BINARY ----
    binary_pred = stage1_model.predict(X)[0]
    if binary_pred == 0:
        st.success("🟢 Stage 1: BENIGN traffic detected")
    else:
        st.error("🔴 Stage 1: MALICIOUS traffic detected")

        # ---- STAGE 2: MULTICLASS ----
        attack_idx = stage2_model.predict(X)[0]
        attack_name = encoder.inverse_transform([attack_idx])[0]
        st.warning(f"Stage 2 Attack Type: **{attack_name}**")
