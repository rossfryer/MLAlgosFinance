# 📈 Stock Market Machine Learning Dashboard

This interactive dashboard demonstrates various machine learning models applied to stock market data. Hosted on Hugging Face Spaces and built with **Streamlit**, the app allows users to explore linear and classification models on a financial dataset including technical indicators, macroeconomic features, and sentiment scores.

## 🚀 Features

- 📊 Visual data exploration and modeling
- 🧠 Select and compare the following ML models:
  - Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- 📈 Visual output of model predictions
- 🧾 Model explanation included
- ⚙️ Dynamic feature and target selection

## 📂 Dataset

The dataset is `stock_market_dataset.csv` and includes:

- Daily stock data (`Open`, `Close`, `High`, `Low`, `Volume`)
- Technical indicators (`RSI`, `MACD`)
- Macroeconomic indicators (`GDP`, `Inflation`)
- Sentiment analysis scores

## 🛠 How to Use

1. Select a **target variable** (e.g. `Close`)
2. Choose one or more **features** (e.g. `Open`, `Volume`)
3. Select the **machine learning model** to apply
4. View prediction results and error/accuracy metrics
5. Read the explanation of how each model works

## 📦 Requirements

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

```
├── app.py                  # Streamlit app
├── stock_market_dataset.csv  # Dataset used in the app
├── requirements.txt        # Required Python libraries
├── README.md               # Project overview and instructions
```

## 📍 Hosted on

This app is live on [Hugging Face Spaces](https://huggingface.co/spaces) – just open it and explore!

---

*Developed with ❤️ using Streamlit and scikit-learn.*
