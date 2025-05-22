import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="Stock Market ML Dashboard", layout="wide")
st.title("📊 Stock Market ML Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("stock_market_dataset.csv")

df = load_data()

# --- User selection ---
st.sidebar.header("Feature Selection")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target = st.sidebar.selectbox("Select Target Variable", options=numeric_cols)
features = st.sidebar.multiselect("Select Feature Columns", options=[col for col in numeric_cols if col != target], default=["Open", "Volume"])

model_type = st.sidebar.selectbox("Select Model", [
    "Linear Regression",
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Logistic Regression",
    "KNN Classification"
])

show_data = st.sidebar.checkbox("Show Data Preview")

if show_data:
    st.write("## Raw Dataset Preview")
    st.dataframe(df.head())

# --- Preprocessing ---
if len(features) == 0 or target not in df.columns:
    st.warning("Please select a valid target and at least one feature.")
    st.stop()

X = df[features]
y = df[target]

# Create classification label if needed
if model_type in ["Logistic Regression", "KNN Classification"]:
    y = (y.shift(-1) > y).astype(int)

# Remove rows with NaN after shifting
data = pd.concat([X, y], axis=1).dropna()
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Model Execution ---
st.write(f"## Running: {model_type}")

if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.write("### Mean Squared Error:", mean_squared_error(y_test, pred))
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": pred}))

elif model_type == "Multiple Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.write("### Mean Squared Error:", mean_squared_error(y_test, pred))
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": pred}))

elif model_type == "Polynomial Regression":
    degree = st.sidebar.slider("Polynomial Degree", min_value=2, max_value=5, value=3)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)
    st.write("### Mean Squared Error:", mean_squared_error(y_test, pred))
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": pred}))

elif model_type == "Logistic Regression":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.write("### Accuracy:", accuracy_score(y_test, pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred))

elif model_type == "KNN Classification":
    k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=15, value=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.write("### Accuracy:", accuracy_score(y_test, pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred))

# --- Explanation Section ---
st.write("---")
st.write("## Model Explanation")

if model_type == "Linear Regression":
    st.markdown("""
    **Linear Regression** attempts to model the relationship between a single feature and a continuous target variable
    using a straight line.
    """)

elif model_type == "Multiple Linear Regression":
    st.markdown("""
    **Multiple Linear Regression** generalizes linear regression to include multiple features, allowing it to model more complex relationships.
    """)

elif model_type == "Polynomial Regression":
    st.markdown("""
    **Polynomial Regression** extends linear regression by adding polynomial terms of the features, enabling it to model nonlinear patterns.
    """)

elif model_type == "Logistic Regression":
    st.markdown("""
    **Logistic Regression** is used for binary classification problems. It outputs a probability and classifies values into one of two categories.
    """)

elif model_type == "KNN Classification":
    st.markdown("""
    **K-Nearest Neighbors (KNN)** is a non-parametric method that classifies data based on the closest training examples in the feature space.
    """)
