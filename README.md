# Network Intrusion Detection Agent

## Project Overview
This project implements an intelligent network intrusion detection system using a multi-stage AI approach:

1. **Unsupervised Anomaly Detection**  
   - Isolation Forest is trained on benign traffic to detect anomalies (potential attacks).
   - Produces an anomaly score for each network flow.

2. **Binary Classification (Stage 1)**  
   - XGBoost and Random Forest models classify traffic as **Benign** or **Malicious**.

3. **Multiclass Attack Classification (Stage 2)**  
   - If traffic is malicious, a second XGBoost and RandomForest model predicts the type of attack (e.g., Exploits, Backdoor, DoS, etc.).

The pipeline is accessible via a **Streamlit web interface** where users can input flow features or generate random samples for testing.


## Installation and Setup

### 1. Clone the Repository
```bash
git clone <https://github.com/Markl1T/network-intrusion-detection-agent>
cd network-intrusion-detector-agent
```

### 2. Download the dataset from [NF-UNSW-NB15-V2 Dataset](https://www.kaggle.com/datasets/sankurisrinath/nf-unsw-nb15-v2csv/data) and put it in /data folder

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Streamlit app
```bash
streamlit run src/app.py
```


## Directory Structure
```
network-intrusion-detector-agent/
│
├─ data/                   # Contains dataset and sampled flows
│   ├─ NF-UNSW-NB15-v2.csv
│   └─ dataset_sample.csv
│
├─ models/                 # Saved AI models (Stage1, Stage2, Anomaly) in pkl files
│
├─ src/
│   ├─ app.py              # Streamlit interface
│   ├─ build_sample.py     # Script to build sample pool
│   ├─ config.py           # Paths, features, and constants
│   ├─ preprocessing.py    # Feature cleaning and encoding
│   ├─ train_anomaly.py    # Train the model to detect anomalies
│   ├─ train_stage1.py     # Train the model to classify Benign/Malicious classification
│   └─ train_stage2.py     # Train the model to classify the type of malicious attack
│
├─ requirements.txt        # Requirements to run the code
└─ README.md
```


## AI Methodologies

### Unsupervised Learning
 - **Isolation Forest**
   - Trained only on benign flows to detect anomalous network behavior without requiring labeled attacks.

### Supervised Learning
 - **Binary Classifier with XGBoost/RandomForest**
   - Classifies traffic as Benign or Malicious based on labeled dataset features.

 - **Multiclass Classifier with XGBoost/RandomForest**
   - Predicts specific attack type using malicious flow features. Encodes the attack labels using a LabelEncoder.


## Notes

Streamlit interface uses only the most important features for user input, other features are filled automatically for model compatibility.

Random flows are sampled from a pool of benign and malicious traffic to simulate realistic testing.