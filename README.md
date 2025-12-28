# 🚨 Network Anomaly Detection Web App
#### **A deep learning-powered Flask application that detects DDoS attacks and HTTP anomalies in network traffic. The app supports CSV file uploads, generates predictions using trained LSTM**models, and visualizes results in a modern, animated interface**

![i](images.png)

.
```

.
├── app.py                     # Main Flask app
├── models/
│   ├── best_model-ddos.h5     # LSTM model for DDoS detection
│   ├── best_model-cic.h5      # LSTM model for CIC log data
│   └── scaler.pkl             # Pre-trained scaler (joblib)
├── templates/
│   ├── index.html             # Home page
│   ├── model1.html            # DDoS prediction page
│   └── model2.html            # CIC prediction page
├── static/
│   └── css/
│       └── style.css          # Custom dark theme + animations
├── page_mapping.json          # Encoded page label mapping
├── requirements.txt           # All Python dependencies
└── README.md                  # You're here!
