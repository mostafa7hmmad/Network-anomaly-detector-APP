
import streamlit as st
import pandas as pd
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gdown
import json
from tensorflow.keras.models import load_model
import streamlit as st

def run():
    st.header("CIC DDoS Detection App")

    @st.cache_resource
    def load_resources():
        # ✅ Replace these with the actual Google Drive file IDs
        model_file_id = "1RIWyICckTLH5pgLaEkDbEumSTTFG5K9B"   # For best_model-cic.h5
        mapping_file_id = "1rc6fO1DpXKTIcWr4PHm-mNiCahMYcm4L"  # For page_mapping.json

        # 🔽 Download model if not exists
        if not os.path.exists("best_model-cic.h5"):
            model_url = f"https://drive.google.com/uc?id={model_file_id}"
            gdown.download(model_url, "best_model-cic.h5", quiet=False)

        # 🔽 Download mapping file if not exists
        if not os.path.exists("page_mapping.json"):
            mapping_url = f"https://drive.google.com/uc?id={mapping_file_id}"
            gdown.download(mapping_url, "page_mapping.json", quiet=False)

        model = load_model("best_model-cic.h5")
        with open("page_mapping.json", "r") as f:
            page_map = json.load(f)

        return model, page_map

    # Call it
    model, page_map = load_resources()


    def preprocess(df):
        df = df.copy()
        df["Method"] = df["Method"].map({"GET":0, "POST":1, "PUT":2})
        df["lenght"] = df["Content-Length"].str.extract(r'Content-Length:\s*(\d+)').astype(float)
        df["page"] = df["URL"].str.extract(r'/([^/]+\.jsp)')
        df["clean_URL"] = df["URL"].str.extract(r'tienda1/(.*) HTTP')[0]
        df["clean_URL"] = LabelEncoder().fit_transform(df["clean_URL"])
        df["page_encoded"] = df["page"].map(page_map)
        tok = Tokenizer(); tok.fit_on_texts(df["cookie"])
        seq = tok.texts_to_sequences(df["cookie"])
        pad = pad_sequences(seq)
        df["cookie_1"] = pad[:,1] if pad.shape[1] > 1 else pad[:,0]
        scaler = MinMaxScaler()
        cols = ["lenght","clean_URL","cookie_1"]
        df[cols] = scaler.fit_transform(df[cols])
        X = df[["Method","lenght","clean_URL","page_encoded","cookie_1"]].values
        return X.reshape((X.shape[0], 1, 5))

    with st.form("cic_manual"):
        st.subheader("Manual Input (CIC)")
        c1, c2, c3, c4 = st.columns(4)
        m = c1.selectbox("Method", ["GET","POST","PUT"])
        l = c2.text_input("Content-Length (e.g. 'Content-Length: 63')")
        u = c3.text_input("URL (full request line)")
        co = c4.text_input("Cookie string")
        ok = st.form_submit_button("Predict CIC")

    if ok:
        d = pd.DataFrame([{"Method":m, "Content-Length":l, "URL":u, "cookie":co}])
        X = preprocess(d)
        p = model.predict(X)[0][0]
        label = "Normal" if p >= 0.5 else "Anomalous"
        st.write(f"**CIC Prediction:** {label} (Probability: {p:.4f})")

    st.subheader("Batch Testing via CSV")
    up = st.file_uploader("Upload CIC CSV", type=["csv"], key="up_cic")
    if up:
        dfb = pd.read_csv(up)
        Xb = preprocess(dfb)
        ps = model.predict(Xb).flatten()
        labs = ["Normal" if x > 0.55 else "Anomalous" for x in ps]
        cnt = pd.Series(labs).value_counts()
        st.bar_chart(cnt)
        st.write(cnt)
