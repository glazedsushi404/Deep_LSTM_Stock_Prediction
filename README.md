This project leverages **time series analysis** and **deep learning (LSTM)** to predict stock prices using historical financial market data. It incorporates preprocessing of OHLCV (Open, High, Low, Close, Volume) data, computes key technical indicators, and builds a sequential neural network for predictive modeling.

---

## 📊 Dataset

- Source: Historical stock prices  
- Format: CSV-like tabular data with 5060 rows × 5 columns  
- Fields:
  - `date`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`

The dataset spans from **2005-01-03** to **2025-07-18**, containing OHLCV price data.

---

## ⚙️ Features Engineered

Additional financial indicators were computed to augment the dataset:
- **SMA_5**: 5-day Simple Moving Average
- **SMA_20**: 20-day Simple Moving Average
- **RSI**: Relative Strength Index

These features help capture market trends and momentum.

---

## 🧠 Model Architecture

A **Keras Sequential Model** with the following layers:

| Layer Type | Output Shape | Parameters |
|------------|--------------|------------|
| LSTM (64 units)   | (None, 1, 64) | 17,664 |
| Dropout           | (None, 1, 64) | 0 |
| LSTM (32 units)   | (None, 32)    | 12,416 |
| Dropout           | (None, 32)    | 0 |
| Dense (1 unit)    | (None, 1)     | 33 |

- **Total Trainable Parameters**: 30,113  
- **Dropout layers** are used for regularization.

---

## 🧪 Training & Evaluation

The model is trained on a time-series dataset using a sliding window approach. Specific configurations such as:
- **Loss Function**: MSE or MAE (assumed unless otherwise stated)
- **Optimizer**: Adam (default for LSTM-based models)
- **Splitting**: Implicit train-test using recent data for validation

---

