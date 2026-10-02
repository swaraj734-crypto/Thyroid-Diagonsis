import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import encode_single_sample
from src.hybrid_model import ThyroidHybridModel

MODEL_DIR = "models"

st.set_page_config(
    page_title="Thyroid Diagnosis Assistant",
    page_icon="🧬",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = ThyroidHybridModel.load(MODEL_DIR)
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    return model, encoders


def render_header():
    st.title("🧬 Thyroid Diagnosis Assistant")
    st.caption(
        "A CatBoost + ANN hybrid ensemble that screens thyroid function "
        "from standard lab markers. Built as a learning project — "
        "not a substitute for professional medical advice."
    )


def patient_form():
    tab1, tab2 = st.tabs(["Patient Info", "Lab Values"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=110, value=45)
        with col2:
            sex = st.selectbox("Sex", ["F", "M"])

        col3, col4 = st.columns(2)
        with col3:
            on_thyroxine = st.checkbox("Currently on thyroxine therapy?")
        with col4:
            query_hypothyroid = st.checkbox("Clinically suspected hypothyroid?")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            tsh = st.number_input("TSH (µIU/mL)", 0.0, 200.0, 2.0, step=0.1)
            t3 = st.number_input("T3 (nmol/L)", 0.0, 10.0, 1.8, step=0.1)
            tt4 = st.number_input("TT4 (total T4)", 0.0, 300.0, 110.0, step=1.0)
        with col2:
            t4u = st.number_input("T4U (T4 uptake)", 0.0, 2.0, 1.0, step=0.01)
            fti = st.number_input("FTI (Free Thyroxine Index)", 0.0, 300.0, 110.0, step=1.0)

    return {
        "age": age,
        "sex": sex,
        "on_thyroxine": int(on_thyroxine),
        "query_hypothyroid": int(query_hypothyroid),
        "TSH": tsh,
        "T3": t3,
        "TT4": tt4,
        "T4U": t4u,
        "FTI": fti,
    }


def risk_meter(label: str, probability: float):
    st.write(f"**{label}**")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"{probability * 100:.1f}% confidence")


def main():
    render_header()

    if not os.path.exists(os.path.join(MODEL_DIR, "hybrid_meta.pkl")):
        st.warning(
            "No trained model found yet. Run `python train_model.py` "
            "first to train and save the hybrid model."
        )
        return

    model, encoders = load_artifacts()
    sample = patient_form()

    if st.button("Run Diagnosis", type="primary"):
        X_sample = encode_single_sample(sample, encoders)
        probs = model.predict_proba(X_sample.values)[0]

        class_names = encoders["target"].inverse_transform(
            np.arange(len(probs))
        )

        st.subheader("Result")
        top_idx = int(np.argmax(probs))
        st.success(f"Most likely category: **{class_names[top_idx]}**")

        st.divider()
        st.write("### Confidence breakdown")
        for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            risk_meter(name.capitalize(), float(p))

        st.divider()
        with st.expander("Which markers mattered most?"):
            st.write(
                "TSH and FTI are typically the strongest signals for "
                "thyroid dysfunction — a high TSH with a low FTI usually "
                "points toward hypothyroidism, while a suppressed TSH "
                "with an elevated FTI points toward hyperthyroidism."
            )
            st.dataframe(pd.DataFrame([sample]))

    st.divider()
    st.caption(
        "⚠️ Educational project only. All results should be verified "
        "by a certified medical professional before any clinical use."
    )


if __name__ == "__main__":
