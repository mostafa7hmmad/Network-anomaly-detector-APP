# app.py
from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.activations import sigmoid

app = Flask(__name__)

# --- Load models and resources ---
model1 = load_model("models/best_model-ddos.h5")
scaler1 = joblib.load("models/scaler.pkl")
model2 = load_model("models/best_model-cic.h5")
with open("models/page_mapping.json") as f:
    page_map = json.load(f)

# --- Utilities ---
features_model1 = [
    'Bwd_Packet_Length_Std', 'Bwd_Packet_Length_Max', 'Avg_Bwd_Segment_Size', 'Bwd_Packet_Length_Mean',
    'Total_Length_of_Bwd_Packets', 'Packet_Length_Variance', 'Average_Packet_Size', 'Packet_Length_Std',
    'Max_Packet_Length', 'Destination_Port', 'Subflow_Bwd_Bytes', 'Packet_Length_Mean',
    'Subflow_Fwd_Packets', 'Bwd_Header_Length', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Flow_IAT_Std', 'Subflow_Bwd_Packets', 'Fwd_IAT_Std', 'Fwd_Header_Length', 'Flow_IAT_Max', 'Idle_Min',
    'Fwd_Header_Length1', 'Flow_IAT_Mean', 'Subflow_Fwd_Bytes', 'Fwd_Packet_Length_Max',
    'Init_Win_bytes_forward', 'Total_Length_of_Fwd_Packets', 'Fwd_IAT_Total', 'Avg_Fwd_Segment_Size']

def scale_input(X):
    if X.ndim == 3:
        n_samples, timesteps, _ = X.shape
        X_flat = X.reshape(n_samples, timesteps)
        X_scaled = scaler1.transform(X_flat)
        return X_scaled.reshape(n_samples, timesteps, 1)
    return scaler1.transform(X)

def preprocess_model2(df):
    df = df.copy()
    df["Method"] = df["Method"].map({"GET": 0, "POST": 1, "PUT": 2})
    df["lenght"] = df["Content-Length"].str.extract(r'Content-Length:\s*(\d+)').astype(float)
    df["page"] = df["URL"].str.extract(r'/([^/]+\.jsp)')
    df["clean_URL"] = df["URL"].str.extract(r'tienda1/(.*) HTTP')[0]
    df["clean_URL"] = LabelEncoder().fit_transform(df["clean_URL"].astype(str))
    df["page_encoded"] = df["page"].map(page_map)
    tok = Tokenizer(); tok.fit_on_texts(df["cookie"].astype(str))
    seq = tok.texts_to_sequences(df["cookie"].astype(str))
    pad = pad_sequences(seq)
    df["cookie_1"] = pad[:,1] if pad.shape[1] > 1 else pad[:,0]
    scaler = MinMaxScaler()
    cols = ["lenght", "clean_URL", "cookie_1"]
    df[cols] = scaler.fit_transform(df[cols])
    X = df[["Method", "lenght", "clean_URL", "page_encoded", "cookie_1"]].values
    return X.reshape((X.shape[0], 1, 5))

def create_seaborn_plot(results):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, values = zip(*results.items())
    sns.barplot(x=list(labels), y=list(values), palette="cool", ax=ax)
    ax.set_title("Prediction Counts", color='white')
    ax.set_xlabel("Label", color='white')
    ax.set_ylabel("Count", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.read()).decode('utf8')
    plt.close(fig)
    return f'data:image/png;base64,{plot_url}'

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1', methods=['GET', 'POST'])
def model1_view():
    results = None
    plot_img = None
    model_name = "Detox Model"
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df = df[features_model1]
            df = df.astype({col: float for col in df.columns})
            X = df.values.reshape((-1, len(features_model1), 1))
            X_scaled = scale_input(X)
            raw_probs = model1.predict(X_scaled).flatten()
            probs = sigmoid(raw_probs).numpy()
            preds = ['Benign' if p > 0.55 else 'Attack' for p in probs]
            results = dict(pd.Series(preds).value_counts())
            plot_img = create_seaborn_plot(results)
    return render_template('model1.html', results=results, model_name=model_name, plot_img=plot_img)

@app.route('/model2', methods=['GET', 'POST'])
def model2_view():
    results = None
    plot_img = None
    model_name = "CIC Model"
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            X = preprocess_model2(df)
            ps = model2.predict(X).flatten()
            labs = ["Normal" if x > 0.55 else "Anomalous" for x in ps]
            results = dict(pd.Series(labs).value_counts())
            plot_img = create_seaborn_plot(results)
    return render_template('model2.html', results=results, model_name=model_name, plot_img=plot_img)

if __name__ == '__main__':
    app.run(debug=True)
