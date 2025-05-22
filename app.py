import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings

st.set_page_config(page_title="Stock Market ML Dashboard", layout="wide")
st.title("📊 Stock Market ML Dashboard")

# Introduction and Instructions
st.markdown("""
## 🎓 Learning Machine Learning with Stock Market Data

This interactive dashboard helps you learn machine learning concepts using real stock market data. Below are some suggested exercises to try:

### 📈 Exercise 1: Price Prediction
**Goal**: Predict the next day's closing price
1. Select "Next_Close" as your target variable
2. Choose these features: `SMA_10`, `RSI`, `MACD`, `Close`, `Volume`
3. Try different models:
   - Start with "Linear Regression" to understand basic relationships
   - Move to "Multiple Linear Regression" to see how multiple features work together
   - Finally, try "Polynomial Regression" to capture non-linear patterns

### 📊 Exercise 2: Market Direction Classification
**Goal**: Predict whether the price will go up or down
1. Select "Target" as your target variable
2. Choose these features: `RSI`, `MACD`, `Close`, `GDP_Growth`, `Sentiment_Score`
3. Try these models:
   - "Logistic Regression" for a basic classification approach
   - "KNN Classification" to see how similar historical patterns can predict future movements

### 📉 Exercise 3: Volatility Analysis
**Goal**: Understand price volatility
1. Select "High" as your target
2. Choose these features: `Bollinger_Upper`, `Bollinger_Lower`, `Volume`, `Interest_Rate`
3. Use "Polynomial Regression" to capture complex relationships

### 💡 Learning Tips:
- Start with simpler models and fewer features
- Use "Show Data Preview" to understand your data
- Compare model performance using the metrics provided
- Try different feature combinations to see how they affect predictions
- For KNN, experiment with different numbers of neighbors

### 📚 Key Concepts to Explore:
- How do technical indicators affect predictions?
- What's the impact of market sentiment on price movements?
- How do different models handle the same data?
- What's the relationship between model complexity and performance?

---
""")

@st.cache_data
def load_data():
    return pd.read_csv("stock_market_dataset.csv")

df = load_data()

# --- User selection ---
st.sidebar.header("Feature Selection")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target = st.sidebar.selectbox("Select Target Variable", options=numeric_cols)
features = st.sidebar.multiselect(
    "Select Feature Columns",
    options=[col for col in numeric_cols if col != target]
)
model_type = st.sidebar.selectbox("Select Model", [
    "Simple Linear Regression",
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

# Add model explanations
model_explanations = {
    "Simple Linear Regression": """
    **Simple Linear Regression** models the relationship between a single feature and a continuous target variable.
    - Best for: Understanding the direct relationship between one feature and the target
    - Key concept: The model finds a straight line that best fits the data
    - Interpretation: The slope shows how much the target changes with a unit change in the feature
    """,
    
    "Multiple Linear Regression": """
    **Multiple Linear Regression** extends linear regression to include multiple features.
    - Best for: Analyzing how multiple factors affect the target variable
    - Key concept: Each feature has its own coefficient, showing its individual impact
    - Interpretation: The model shows how different features work together to predict the target
    """,
    
    "Polynomial Regression": """
    **Polynomial Regression** allows for non-linear relationships between features and target.
    - Best for: Capturing curved relationships in the data
    - Key concept: The model can fit curved lines to the data
    - Interpretation: Higher degrees can capture more complex patterns, but beware of overfitting
    """,
    
    "Logistic Regression": """
    **Logistic Regression** is used for binary classification problems.
    - Best for: Predicting whether something will happen or not
    - Key concept: Outputs probabilities between 0 and 1
    - Interpretation: The model estimates the probability of the target being in a particular class
    """,
    
    "KNN Classification": """
    **K-Nearest Neighbors (KNN)** classifies based on similar historical patterns.
    - Best for: Finding patterns in data based on similar historical cases
    - Key concept: Uses the k most similar historical cases to make predictions
    - Interpretation: The model looks at what happened in similar situations in the past
    """
}

st.markdown(model_explanations[model_type])

# Add detailed algorithm explanations
def show_algorithm_details(model_type):
    with st.expander("📚 Detailed Algorithm Explanation"):
        if model_type == "Simple Linear Regression":
            st.markdown("""
            ### Simple Linear Regression: Mathematical Foundation
            
            Simple linear regression models the relationship between a single feature (x) and a target variable (y) using a straight line:
            
            $$y = \\beta_0 + \\beta_1x + \\epsilon$$
            
            Where:
            - $\\beta_0$ is the y-intercept (where the line crosses the y-axis)
            - $\\beta_1$ is the slope (how much y changes for a unit change in x)
            - $\\epsilon$ is the error term
            
            The model finds the best line by minimizing the sum of squared errors:
            
            $$\\min_{\\beta_0, \\beta_1} \\sum_{i=1}^{n} (y_i - (\\beta_0 + \\beta_1x_i))^2$$
            
            The optimal coefficients are found using:
            
            $$\\beta_1 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}$$
            
            $$\\beta_0 = \\bar{y} - \\beta_1\\bar{x}$$
            
            Where $\\bar{x}$ and $\\bar{y}$ are the means of x and y respectively.
            
            ### Visual Explanation
            """)
            
            # Create a simple visualization of linear regression
            x = np.linspace(-10, 10, 100)
            y = 2*x + 1 + np.random.normal(0, 2, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
            
            # Add the true line
            fig.add_trace(go.Scatter(x=x, y=2*x + 1, mode='lines', name='True Relationship'))
            
            # Add the fitted line
            slope, intercept = np.polyfit(x, y, 1)
            fig.add_trace(go.Scatter(x=x, y=slope*x + intercept, mode='lines', name='Fitted Line'))
            
            fig.update_layout(
                title="Simple Linear Regression Concept",
                xaxis_title="Feature (X)",
                yaxis_title="Target (y)",
                showlegend=True
            )
            st.plotly_chart(fig)
            
            st.markdown("""
            ### Understanding the Parameters
            
            1. **Slope ($\\beta_1$)**:
               - Represents the rate of change of y with respect to x
               - Positive slope: y increases as x increases
               - Negative slope: y decreases as x increases
               - Larger absolute value: steeper relationship
            
            2. **Y-intercept ($\\beta_0$)**:
               - The value of y when x = 0
               - Where the regression line crosses the y-axis
            
            ### Model Assumptions
            
            1. **Linearity**: The relationship between x and y is linear
            2. **Independence**: Observations are independent of each other
            3. **Homoscedasticity**: The variance of errors is constant
            4. **Normality**: The errors are normally distributed
            
            ### Goodness of Fit
            
            The model's fit is often measured using:
            
            - **R-squared**: Proportion of variance in y explained by x
            - **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values
            - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as y
            """)
            
        elif model_type == "Multiple Linear Regression":
            st.markdown("""
            ### Multiple Linear Regression: Mathematical Foundation
            
            Multiple linear regression models the relationship between a dependent variable (y) and multiple independent variables (X) using a linear equation:
            
            $$y = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n + \\epsilon$$
            
            Where:
            - $\\beta_0$ is the y-intercept
            - $\\beta_1, \\beta_2, ..., \\beta_n$ are the coefficients for each feature
            - $x_1, x_2, ..., x_n$ are the features
            - $\\epsilon$ is the error term
            
            The model finds the best coefficients by minimizing the sum of squared errors:
            
            $$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - (\\beta_0 + \\beta_1x_{i1} + ... + \\beta_px_{ip}))^2$$
            
            This is solved using matrix operations:
            
            $$\\beta = (X^TX)^{-1}X^Ty$$
            
            Where:
            - $X$ is the feature matrix
            - $y$ is the target vector
            - $\\beta$ is the coefficient vector
            
            ### Feature Importance Visualization
            """)
            
            # Create a sample feature importance plot
            features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
            importance = np.random.normal(0.5, 0.2, len(features))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=features,
                y=importance,
                text=[f'{imp:.2f}' for imp in importance],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Feature Importance in Multiple Linear Regression",
                xaxis_title="Features",
                yaxis_title="Coefficient Value",
                showlegend=False
            )
            st.plotly_chart(fig)
            
            st.markdown("""
            ### Understanding the Coefficients
            
            Each coefficient ($\\beta_i$) represents:
            - The expected change in the target variable for a one-unit change in the corresponding feature
            - The impact of that feature while holding all other features constant
            - The relative importance of each feature in predicting the target
            
            ### Model Assumptions
            
            1. Linearity: The relationship between features and target is linear
            2. Independence: Observations are independent of each other
            3. Homoscedasticity: The variance of errors is constant
            4. Normality: The errors are normally distributed
            5. No multicollinearity: Features are not highly correlated with each other
            """)
            
        elif model_type == "Polynomial Regression":
            st.markdown("""
            ### Polynomial Regression: Mathematical Foundation
            
            Polynomial regression extends linear regression by adding polynomial terms:
            
            $$y = \\beta_0 + \\beta_1x + \\beta_2x^2 + ... + \\beta_nx^n + \\epsilon$$
            
            The model can capture non-linear relationships while still using linear regression techniques by transforming the features.
            
            ### Polynomial Degree Effect
            """)
            
            # Create visualization of different polynomial degrees
            x = np.linspace(-5, 5, 100)
            y = np.sin(x) + np.random.normal(0, 0.2, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
            
            # Add polynomial fits of different degrees
            for degree in [1, 2, 3, 4]:
                coeffs = np.polyfit(x, y, degree)
                y_fit = np.polyval(coeffs, x)
                fig.add_trace(go.Scatter(x=x, y=y_fit, mode='lines', name=f'Degree {degree}'))
            
            fig.update_layout(
                title="Effect of Polynomial Degree on Fit",
                xaxis_title="Feature (X)",
                yaxis_title="Target (y)",
                showlegend=True
            )
            st.plotly_chart(fig)
            
        elif model_type == "Logistic Regression":
            st.markdown("""
            ### Logistic Regression: Mathematical Foundation
            
            Logistic regression uses the logistic function to model binary outcomes:
            
            $$P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1x_1 + ... + \\beta_nx_n)}}$$
            
            The model finds the best coefficients by maximizing the log-likelihood:
            
            $$\\max_{\\beta} \\sum_{i=1}^{n} [y_i\\log(p_i) + (1-y_i)\\log(1-p_i)]$$
            
            ### Decision Boundary Visualization
            """)
            
            # Create a sample decision boundary visualization
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z = 1 / (1 + np.exp(-(X + Y)))
            
            fig = go.Figure()
            fig.add_trace(go.Contour(
                x=x, y=y, z=Z,
                colorscale='RdBu',
                showscale=True,
                name='Decision Boundary'
            ))
            
            fig.update_layout(
                title="Logistic Regression Decision Boundary",
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                showlegend=True
            )
            st.plotly_chart(fig)
            
        elif model_type == "KNN Classification":
            st.markdown("""
            ### K-Nearest Neighbors: Mathematical Foundation
            
            KNN classifies points based on the majority class of their k nearest neighbors:
            
            $$\\hat{y} = \\text{mode}(y_{i_1}, y_{i_2}, ..., y_{i_k})$$
            
            Where $i_1, i_2, ..., i_k$ are the indices of the k nearest neighbors.
            
            Distance is typically measured using Euclidean distance:
            
            $$d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$$
            
            ### K-Neighbors Visualization
            """)
            
            # Create a sample KNN visualization
            np.random.seed(42)
            X = np.random.randn(100, 2)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            fig = go.Figure()
            
            # Plot points
            for label in [0, 1]:
                mask = y == label
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode='markers',
                    name=f'Class {label}'
                ))
            
            # Plot a test point and its neighbors
            test_point = np.array([0.5, 0.5])
            distances = np.sqrt(np.sum((X - test_point)**2, axis=1))
            k = 5
            nearest_indices = np.argsort(distances)[:k]
            
            fig.add_trace(go.Scatter(
                x=[test_point[0]], y=[test_point[1]],
                mode='markers',
                marker=dict(size=15, symbol='star'),
                name='Test Point'
            ))
            
            # Add lines to nearest neighbors
            for idx in nearest_indices:
                fig.add_trace(go.Scatter(
                    x=[test_point[0], X[idx, 0]],
                    y=[test_point[1], X[idx, 1]],
                    mode='lines',
                    line=dict(dash='dash'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="K-Nearest Neighbors Classification",
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                showlegend=True
            )
            st.plotly_chart(fig)

# Add exercises section after the model explanations
def show_exercises(model_type):
    with st.expander("📝 Model Exploration Exercises"):
        if model_type == "Simple Linear Regression":
            st.markdown("""
            ### Simple Linear Regression Exercises
            
            1. **Basic Relationship Analysis**
               - Select "Close" as your target and "Open" as your feature
               - Observe the R-squared value and regression line
               - Try different features (e.g., "Volume", "High", "Low")
               - Which feature has the strongest linear relationship with "Close"?
            
            2. **Understanding the Slope**
               - Use "Close" as your target and "Open" as your feature
               - Note the slope value from the model
               - What does this tell you about the relationship between opening and closing prices?
               - Try different time periods by adjusting the number of points
            
            3. **Model Assumptions Check**
               - Plot the residuals (differences between actual and predicted values)
               - Check if they appear randomly distributed
               - Look for patterns that might violate the linearity assumption
            
            4. **Prediction Analysis**
               - Use the model to predict the next day's closing price
               - Compare predictions with actual values
               - What factors might explain any large prediction errors?
            """)
            
        elif model_type == "Multiple Linear Regression":
            st.markdown("""
            ### Multiple Linear Regression Exercises
            
            1. **Feature Selection Impact**
               - Start with "Close" as target and "Open" as the only feature
               - Add "Volume" as a second feature
               - Compare R-squared values
               - What does this tell you about the importance of trading volume?
            
            2. **Technical Indicators**
               - Use "Close" as target
               - Select these features: "RSI", "MACD", "SMA_10"
               - Analyze the coefficients
               - Which technical indicator has the strongest impact?
            
            3. **Market Sentiment Analysis**
               - Target: "Close"
               - Features: "Sentiment_Score", "Volume", "RSI"
               - How well does sentiment predict price movements?
               - Compare with and without sentiment data
            
            4. **Model Complexity**
               - Start with 2 features, then add more
               - Observe how R-squared changes
               - When does adding more features stop improving the model?
            """)
            
        elif model_type == "Polynomial Regression":
            st.markdown("""
            ### Polynomial Regression Exercises
            
            1. **Non-linear Patterns**
               - Target: "Close"
               - Feature: "RSI"
               - Try different polynomial degrees (2-5)
               - When does the model start overfitting?
            
            2. **Price Volatility**
               - Target: "High"
               - Feature: "Volume"
               - Compare linear vs polynomial fits
               - What degree best captures the relationship?
            
            3. **Technical Analysis**
               - Target: "Close"
               - Feature: "MACD"
               - Experiment with different degrees
               - How does the polynomial fit compare to linear regression?
            
            4. **Overfitting Analysis**
               - Use multiple features
               - Try different polynomial degrees
               - Compare training and test set performance
               - When does the model start to overfit?
            """)
            
        elif model_type == "Logistic Regression":
            st.markdown("""
            ### Logistic Regression Exercises
            
            1. **Market Direction Prediction**
               - Target: "Target" (price movement direction)
               - Features: "RSI", "MACD", "Volume"
               - What's the model's accuracy?
               - Which features are most important?
            
            2. **Threshold Analysis**
               - Use the same features as above
               - Try different probability thresholds (0.5, 0.6, 0.7)
               - How does this affect precision and recall?
            
            3. **Technical Indicators**
               - Target: "Target"
               - Compare different combinations of technical indicators
               - Which combination gives the best accuracy?
               - What's the trade-off between precision and recall?
            
            4. **Market Regime Analysis**
               - Use "Target" as your target
               - Add "Market_Regime" as a feature
               - How does the model perform in different market conditions?
            """)
            
        elif model_type == "KNN Classification":
            st.markdown("""
            ### KNN Classification Exercises
            
            1. **Neighbor Count Impact**
               - Target: "Target"
               - Features: "RSI", "MACD"
               - Try different values of k (3, 5, 7, 9)
               - What's the optimal number of neighbors?
            
            2. **Feature Scaling**
               - Use the same features as above
               - Compare performance with and without scaling
               - Why does scaling matter in KNN?
            
            3. **Pattern Recognition**
               - Target: "Target"
               - Features: "Close", "Volume", "RSI"
               - How well does KNN identify similar market patterns?
               - Compare with logistic regression
            
            4. **Market Conditions**
               - Use "Target" as your target
               - Add time-based features
               - How does KNN perform in different market conditions?
               - What patterns does it identify?
            """)
        
        st.markdown("""
        ### General Tips for All Exercises:
        
        1. **Data Quality**
           - Check for missing values
           - Look for outliers
           - Consider feature scaling
        
        2. **Model Evaluation**
           - Compare different metrics
           - Use cross-validation when possible
           - Consider both training and test performance
        
        3. **Feature Engineering**
           - Try creating new features
           - Consider feature interactions
           - Remove redundant features
        
        4. **Documentation**
           - Record your findings
           - Note which combinations work best
           - Document any interesting patterns
        """)

# Add data validation function
def validate_data(X, y, model_type):
    if model_type in ["Logistic Regression", "KNN Classification"]:
        # Check for class imbalance
        class_counts = pd.Series(y).value_counts()
        if len(class_counts) < 2:
            st.error("Error: Target variable has only one class. Classification requires at least two classes.")
            return False
        
        # Check class balance
        min_class_ratio = min(class_counts) / max(class_counts)
        if min_class_ratio < 0.1:  # If one class is less than 10% of the other
            st.warning(f"Warning: Severe class imbalance detected. Class ratio: {min_class_ratio:.2f}")
            st.info("Consider using class weights or resampling techniques.")
        
        # Check for sufficient samples
        if len(y) < 10:
            st.error("Error: Not enough samples for classification. Need at least 10 samples.")
            return False
    
    # Check for missing values
    if X.isnull().any().any() or pd.Series(y).isnull().any():
        st.error("Error: Data contains missing values. Please handle missing values before training.")
        return False
    
    return True

# Add the model execution code for simple linear regression
if model_type == "Simple Linear Regression":
    if len(features) > 1:
        st.warning("Simple Linear Regression works with only one feature. Please select just one feature.")
        st.stop()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = model.score(X_test, y_test)
    
    st.write("### Model Performance")
    st.write("#### Mean Squared Error:", mse)
    st.write("#### R-squared:", r2)
    
    st.markdown("""
    **Understanding the Metrics**:
    - MSE: Lower values indicate better predictions
    - R-squared: Higher values (closer to 1) indicate better fit
    """)
    
    # Enhanced visualization with fewer points
    num_points = st.slider("Number of points to display", min_value=50, max_value=len(y_test), value=100, step=50)
    smoothing = st.slider("Smoothing window", min_value=1, max_value=20, value=5, help="Larger values create smoother lines")
    
    # Create smoothed data
    chart_data = pd.DataFrame({
        "Actual Values": y_test.values[-num_points:],
        "Predicted Values": pred[-num_points:]
    }, index=range(num_points))
    
    # Apply smoothing
    chart_data = chart_data.rolling(window=smoothing).mean()
    
    st.write("### Prediction Visualization")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual Values"],
        name="Actual Values",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted Values"],
        name="Predicted Values",
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Values",
        xaxis_title="Time Points",
        yaxis_title="Values",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the regression line
    st.write("### Regression Line")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_test.values.flatten(),
        y=y_test.values,
        mode='markers',
        name='Data Points'
    ))
    fig.add_trace(go.Scatter(
        x=X_test.values.flatten(),
        y=pred,
        mode='lines',
        name='Regression Line'
    ))
    
    fig.update_layout(
        title="Simple Linear Regression Line",
        xaxis_title=features[0],
        yaxis_title=target,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    show_algorithm_details(model_type)
    show_exercises(model_type)

elif model_type == "Multiple Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    st.write("### Model Performance")
    st.write("#### Mean Squared Error:", mse)
    st.markdown("""
    **Understanding Mean Squared Error (MSE)**:
    - Lower values indicate better predictions
    - The value represents the average squared difference between predictions and actual values
    - Compare MSE across different models to see which performs better
    """)
    
    # Enhanced visualization with fewer points
    num_points = st.slider("Number of points to display", min_value=50, max_value=len(y_test), value=100, step=50)
    smoothing = st.slider("Smoothing window", min_value=1, max_value=20, value=5, help="Larger values create smoother lines")
    
    # Create smoothed data
    chart_data = pd.DataFrame({
        "Actual Values": y_test.values[-num_points:],
        "Predicted Values": pred[-num_points:]
    }, index=range(num_points))
    
    # Apply smoothing
    chart_data = chart_data.rolling(window=smoothing).mean()
    
    st.write("### Prediction Visualization")
    # Use plotly for better visualization
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual Values"],
        name="Actual Values",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted Values"],
        name="Predicted Values",
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Values",
        xaxis_title="Time Points",
        yaxis_title="Values",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Reading the Chart**:
    - Blue line: Actual values from the test set
    - Red line: Model's predictions
    - Closer the lines are to each other, better the model's performance
    - Use the sliders above to adjust the number of points and smoothing
    """)

    show_algorithm_details(model_type)
    show_exercises(model_type)

elif model_type == "Polynomial Regression":
    degree = st.sidebar.slider("Polynomial Degree", min_value=2, max_value=5, value=3)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, pred)
    st.write("### Model Performance")
    st.write("#### Mean Squared Error:", mse)
    st.markdown("""
    **Understanding Mean Squared Error (MSE)**:
    - Lower values indicate better predictions
    - The value represents the average squared difference between predictions and actual values
    - Compare MSE across different models to see which performs better
    - For polynomial regression, watch for overfitting as degree increases
    """)
    
    # Enhanced visualization with fewer points
    num_points = st.slider("Number of points to display", min_value=50, max_value=len(y_test), value=100, step=50)
    smoothing = st.slider("Smoothing window", min_value=1, max_value=20, value=5, help="Larger values create smoother lines")
    
    # Create smoothed data
    chart_data = pd.DataFrame({
        "Actual Values": y_test.values[-num_points:],
        "Predicted Values": pred[-num_points:]
    }, index=range(num_points))
    
    # Apply smoothing
    chart_data = chart_data.rolling(window=smoothing).mean()
    
    st.write("### Prediction Visualization")
    # Use plotly for better visualization
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual Values"],
        name="Actual Values",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted Values"],
        name="Predicted Values",
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Values",
        xaxis_title="Time Points",
        yaxis_title="Values",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Reading the Chart**:
    - Blue line: Actual values from the test set
    - Red line: Model's predictions
    - Closer the lines are to each other, better the model's performance
    - Watch for overfitting: if the red line is too wiggly, try reducing the polynomial degree
    - Use the sliders above to adjust the number of points and smoothing
    """)

    show_algorithm_details(model_type)
    show_exercises(model_type)

elif model_type == "Logistic Regression":
    if not validate_data(X, y, model_type):
        st.stop()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    model = LogisticRegression(class_weight='balanced')  # Add class weights
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics with zero_division parameter
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    st.write("### Model Performance")
    st.write("#### Accuracy:", accuracy)
    st.write("#### Precision:", precision)
    st.write("#### Recall:", recall)
    st.write("#### F1 Score:", f1)
    
    st.markdown("""
    **Understanding Classification Metrics**:
    - Accuracy: Percentage of correct predictions
    - Precision: How many of the predicted positive cases were actually positive
    - Recall: How many of the actual positive cases were correctly identified
    - F1-score: Harmonic mean of precision and recall
    """)
    
    # Enhanced visualization with fewer points
    num_points = st.slider("Number of points to display", min_value=50, max_value=len(y_test), value=100, step=50)
    smoothing = st.slider("Smoothing window", min_value=1, max_value=20, value=5, help="Larger values create smoother lines")
    
    # Create smoothed data
    chart_data = pd.DataFrame({
        "Actual Class": y_test.values[-num_points:],
        "Predicted Probability": pred_proba[-num_points:]
    }, index=range(num_points))
    
    # Apply smoothing
    chart_data = chart_data.rolling(window=smoothing).mean()
    
    st.write("### Classification Visualization")
    # Use plotly for better visualization
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual Class"],
        name="Actual Class",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted Probability"],
        name="Predicted Probability",
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Classification Results",
        xaxis_title="Time Points",
        yaxis_title="Class/Probability",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Reading the Chart**:
    - Blue line: Actual class (0 or 1)
    - Red line: Model's predicted probability (0 to 1)
    - When red line is above 0.5, model predicts class 1
    - When red line is below 0.5, model predicts class 0
    - Use the sliders above to adjust the number of points and smoothing
    """)
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred, zero_division=0))

    show_algorithm_details(model_type)
    show_exercises(model_type)

elif model_type == "KNN Classification":
    if not validate_data(X, y, model_type):
        st.stop()
    
    k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=15, value=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')  # Use distance-based weights
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics with zero_division parameter
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    st.write("### Model Performance")
    st.write("#### Accuracy:", accuracy)
    st.write("#### Precision:", precision)
    st.write("#### Recall:", recall)
    st.write("#### F1 Score:", f1)
    
    st.markdown("""
    **Understanding Classification Metrics**:
    - Accuracy: Percentage of correct predictions
    - Precision: How many of the predicted positive cases were actually positive
    - Recall: How many of the actual positive cases were correctly identified
    - F1-score: Harmonic mean of precision and recall
    
    **KNN Specific Tips**:
    - Try different values of k to find the optimal number of neighbors
    - Too few neighbors can lead to overfitting
    - Too many neighbors can lead to underfitting
    - Distance-based weights can help with imbalanced classes
    """)
    
    # Enhanced visualization with fewer points
    num_points = st.slider("Number of points to display", min_value=50, max_value=len(y_test), value=100, step=50)
    smoothing = st.slider("Smoothing window", min_value=1, max_value=20, value=5, help="Larger values create smoother lines")
    
    # Create smoothed data
    chart_data = pd.DataFrame({
        "Actual Class": y_test.values[-num_points:],
        "Predicted Probability": pred_proba[-num_points:]
    }, index=range(num_points))
    
    # Apply smoothing
    chart_data = chart_data.rolling(window=smoothing).mean()
    
    st.write("### Classification Visualization")
    # Use plotly for better visualization
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Actual Class"],
        name="Actual Class",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Predicted Probability"],
        name="Predicted Probability",
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Classification Results",
        xaxis_title="Time Points",
        yaxis_title="Class/Probability",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Reading the Chart**:
    - Blue line: Actual class (0 or 1)
    - Red line: Model's predicted probability (0 to 1)
    - When red line is above 0.5, model predicts class 1
    - When red line is below 0.5, model predicts class 0
    - Try adjusting k to make the red line more stable
    - Use the sliders above to adjust the number of points and smoothing
    """)
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred, zero_division=0))

    show_algorithm_details(model_type)
    show_exercises(model_type)

# --- Explanation Section ---
st.write("---")
st.write("## Model Explanation")

if model_type == "Simple Linear Regression":
    st.markdown("""
    **Simple Linear Regression** models the relationship between a single feature and a continuous target variable.
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
