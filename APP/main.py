# app_main.py
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Import sub‑app modules
import appddos
import appcic

# Must be first Streamlit command
st.set_page_config(
    page_title="Network Traffic Anomaly Detection Suite",
    layout="wide",
)

# Load & display a Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json")
if lottie:
    st.header("Network Traffic Anomaly Detection Suite")
    st_lottie(lottie, height=150)
else:
    st.title("Network Traffic Anomaly Detection Suite")

# Tabs for each sub‑app
tab1, tab2 = st.tabs(["DDoS Attacks App", "CIC DDoS Detection App"])

with tab1:
    appddos.run()

with tab2:
    appcic.run()
