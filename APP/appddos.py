

import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import sigmoid
from streamlit_lottie import st_lottie
import os
import gdown

def run():
    # Set page layout to wide for full-width form
    st.header("DDoS Attacks App")  # Title above animation

    # Load Lottie animation
    # def load_lottie_url(url):
    #     r = requests.get(url)
    #     if r.status_code != 200:
    #         return None
    #     return r.json()

    # lottie_animation = load_lottie_url(
    #     "https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json"
    # )
    # if lottie_animation:
    #     st_lottie(lottie_animation, height=200, key="traffic_animation")

    # Load the saved model and scaler
    



    @st.cache_resource
    def load_resources():
        # ✅ Replace with actual file IDs from your Drive
        model_file_id = "1A0fie_M6MNVmL_fHxHKWFNwYwwrbdUbm"    # ID for best_model-ddos.h5
        scaler_file_id = "1zM07bEj8NUqIQmfs_iS7K9Bsl3YvmVUd"   # ID for scaler.pkl

        # 🔽 Download model if not exists
        if not os.path.exists("best_model-ddos.h5"):
            model_url = f"https://drive.google.com/uc?id={model_file_id}"
            gdown.download(model_url, "best_model-ddos.h5", quiet=False)

        # 🔽 Download scaler if not exists
        if not os.path.exists("scaler.pkl"):
            scaler_url = f"https://drive.google.com/uc?id={scaler_file_id}"
            gdown.download(scaler_url, "scaler.pkl", quiet=False)

        # ✅ Load resources
        model = load_model("best_model-ddos.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler


    model, scaler = load_resources()

    # Utility to scale 2D or 3D arrays
    def scale_input(X):
        if X.ndim == 3:
            n_samples, timesteps, _ = X.shape
            X_flat = X.reshape(n_samples, timesteps)
            X_scaled = scaler.transform(X_flat)
            return X_scaled.reshape(n_samples, timesteps, 1)
        return scaler.transform(X)

    # Define feature list and their dtypes (30 features matching training shape)
    features = [
        ('Bwd_Packet_Length_Std', float), ('Bwd_Packet_Length_Max', int),
        ('Avg_Bwd_Segment_Size', float), ('Bwd_Packet_Length_Mean', float),
        ('Total_Length_of_Bwd_Packets', int), ('Packet_Length_Variance', float),
        ('Average_Packet_Size', float), ('Packet_Length_Std', float),
        ('Max_Packet_Length', int), ('Destination_Port', int),
        ('Subflow_Bwd_Bytes', int), ('Packet_Length_Mean', float),
        ('Subflow_Fwd_Packets', int), ('Bwd_Header_Length', int),
        ('Total_Fwd_Packets', int), ('Total_Backward_Packets', int),
        ('Flow_IAT_Std', float), ('Subflow_Bwd_Packets', int),
        ('Fwd_IAT_Std', float), ('Fwd_Header_Length', int),
        ('Flow_IAT_Max', int), ('Idle_Min', int),
        ('Fwd_Header_Length1', int), ('Flow_IAT_Mean', float),
        ('Subflow_Fwd_Bytes', int), ('Fwd_Packet_Length_Max', int),
        ('Init_Win_bytes_forward', int), ('Total_Length_of_Fwd_Packets', int),
        ('Fwd_IAT_Total', int), ('Avg_Fwd_Segment_Size', float)
    ]

    st.title("LSTM Network Traffic Model Tester")
    st.markdown(
        f"This app matches your training reshape: (samples, timesteps={len(features)}, 1 feature)"
    )

    # Manual input form: arranged in 4 columns across full width
    with st.form("manual_input_form"):
        st.subheader(f"Manual Input (Full Sequence of length {len(features)})")
        input_data = {}
        for i in range(0, len(features), 4):
            cols = st.columns(4)
            for j, (name, dtype) in enumerate(features[i:i + 4]):
                with cols[j]:
                    if dtype is int:
                        input_data[name] = st.number_input(name, value=0, step=1, key=name)
                    else:
                        input_data[name] = st.number_input(name, value=0.0, format="%.6f", key=name)
        submitted = st.form_submit_button("Predict Single Sequence")

    if submitted:
        df_input = pd.DataFrame([input_data])
        for col, dtype in features:
            df_input[col] = df_input[col].astype(dtype)
        X = df_input.values.reshape((1, len(features), 1))
        X_scaled = scale_input(X)
        raw_output = model.predict(X_scaled)[0][0]
        prob = sigmoid(raw_output).numpy() if hasattr(raw_output, 'numpy') else float(sigmoid(raw_output))
        label = "Benign" if prob >= 0.5 else "Attack"
        st.write(f"**Prediction:** {label} (Probability: {prob:.4f})")

    # Batch testing via CSV upload
    st.subheader("Batch Testing via CSV Upload")
    uploaded_file = st.file_uploader(
        f"Upload CSV: extra columns ignored, must include at least the known {len(features)} features",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.filter(items=[name for name, _ in features])
        for col, dtype in features:
            df[col] = df[col].astype(dtype)
        X = df.values.reshape((-1, len(features), 1))
        X_scaled = scale_input(X)
        raw_probs = model.predict(X_scaled).flatten()
        probs = sigmoid(raw_probs).numpy() if hasattr(sigmoid(raw_probs), 'numpy') else sigmoid(raw_probs)
        preds = ['Benign' if p >0.55 else 'Attack' for p in probs]
        counts = pd.Series(preds).value_counts()
        st.write("### Prediction Counts")
        st.bar_chart(counts)
        st.write(counts)
