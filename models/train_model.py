import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import metrics

def main():
    print("=== Human Development Index (HDI) Predictor - Training Pipeline ===")
    
    # Define file paths
    dataset_path = os.path.join("dataset", "hdi_dataset.csv")
    model_output_path = "model.pkl"
    
    # 1. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_data.py first.")
        
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    # 2. Preprocessing
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    final_len = len(df)
    if initial_len > final_len:
        print(f"Removed {initial_len - final_len} duplicate records.")
    else:
        print("No duplicate records found.")
        
    # Handle missing values using mean imputation
    # Features
    features = [
        "Life Expectancy", 
        "Expected Years of Schooling", 
        "Mean Years of Schooling", 
        "GNI Per Capita"
    ]
    target = "HDI Score"
    
    # Check for missing values before imputation
    missing_counts = df[features + [target]].isnull().sum()
    print("Missing values count per column:")
    print(missing_counts)
    
    # Imputation for features
    imputer = SimpleImputer(strategy="mean")
    df[features] = imputer.fit_transform(df[features])
    
    # Imputation for target (just in case)
    if df[target].isnull().sum() > 0:
        target_imputer = SimpleImputer(strategy="mean")
        df[[target]] = target_imputer.fit_transform(df[[target]])
        
    # 3. Split features and target
    X = df[features]
    y = df[target]
    
    # Split into Train/Test sets (80:20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print("\nModel Coefficients:")
    for feature, coef in zip(features, model.coef_):
        print(f"  {feature}: {coef:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    
    # 5. Evaluate model
    y_pred = model.predict(X_test)
    
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n=== Model Evaluation Metrics ===")
    print(f"  R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE):              {mae:.4f}")
    print(f"  Mean Squared Error (MSE):               {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE):         {rmse:.4f}")
    print("================================")
    
    # 6. Save model using pickle
    print(f"Saving serialized model to: {model_output_path}")
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
