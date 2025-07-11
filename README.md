# Network-anomaly-detector-APP


📦 1. Data Collection
    └── Gathered from CIC-DDoS2017 dataset and simulated logs.

🧼 2. Data Preprocessing
    └── Feature selection, normalization (MinMax / StandardScaler),
        label encoding, cookie tokenization.

🧠 3. Model Training
    └── LSTM model for temporal pattern recognition
    └── Separate models for CIC and DDoS feature sets.

📁 4. Model Saving
    └── Models saved as .h5 and scaler as .pkl (using joblib)

🌐 5. Streamlit Interface
    └── Two-tab layout:
         • CIC DDoS Detection
         • General DDoS Attack Detection
    └── Supports manual inputs and batch CSV upload.

☁️ 6. Deployment
    └── Hosted on Streamlit Cloud.
    └── Models downloaded from Google Drive using gdown.
