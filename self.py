import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set a random seed for reproducible results
np.random.seed(42)

# --- Step 1: Load Cleaned Data ---
df = pd.read_csv("cleaned_river_dataset_median.csv")

# Define Features (X) and Target (y)
df = df.drop(['Station_Code', "Monitoring_Location", 'State'], axis=1)
X = df.drop('BOD_Max', axis=1)
y = df['BOD_Max']


#Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Preprocessing Numerical Features
standardscaler = StandardScaler()
X_train_scaled = standardscaler.fit_transform(X_train)
X_test_scaled = standardscaler.transform(X_test)
print("Numerical features scaled.")

# Training the Model
model_full = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=8,
    min_samples_leaf=1,
    bootstrap=False
)

model_full.fit(X_train_scaled, y_train)
print("Model training completed.")

# Evaluating the Model
y_pred_full = model_full.predict(X_test_scaled)
r2_full = r2_score(y_test, y_pred_full)
mse_full = mean_squared_error(y_test, y_pred_full)
mae_full = mean_absolute_error(y_test, y_pred_full)
rmse_full = np.sqrt(mse_full)


#-------------------------------------------------------------
#MODEL 2
importances = model_full.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

top_5_features = feature_importance_df['Feature'].head(5).tolist()
print("Top 5 Features:", top_5_features)

X_train_top5 = X_train[top_5_features]
X_test_top5 = X_test[top_5_features]

#model training - create a separate scaler for top 5 features
standardscaler_top5 = StandardScaler()
X_train_top5_scaled = standardscaler_top5.fit_transform(X_train_top5)
X_test_top5_scaled = standardscaler_top5.transform(X_test_top5)

model_top5 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=8,
    min_samples_leaf=1,
    bootstrap=False
)
model_top5.fit(X_train_top5_scaled, y_train)
print("Top 5 Features Model training completed.")

# Evaluating the Top 5 Features Model
y_pred_top5 = model_top5.predict(X_test_top5_scaled)
r2_top5 = r2_score(y_test, y_pred_top5)
mse_top5 = mean_squared_error(y_test, y_pred_top5)
mae_top5 = mean_absolute_error(y_test, y_pred_top5)
rmse_top5 = np.sqrt(mse_top5)

#Model Comparison
print("\n\n--- Model Comparison ---")
print("-------------------------------------------------")
print("| Metric      | Full Model (17) | Top 5 Model |")
print("|-------------|-----------------|-------------|")
print(f"| R-squared   | {r2_full:15.4f} | {r2_top5:11.4f} |")
print(f"| MAE         | {mae_full:15.4f} | {mae_top5:11.4f} |")
print(f"| RMSE        | {rmse_full:15.4f} | {rmse_top5:11.4f} |")
print("-------------------------------------------------")


# BOD Safety Interpretation Function
def interpret_bod_level(bod_value):
    """Interpret BOD value based on safety thresholds"""
    if bod_value <= 3:
        status = "[SAFE] Safe"
        description = "Clean, safe for aquatic life and human use (after treatment)."
    elif bod_value <= 5:
        status = "[CAUTION] Caution"
        description = "Some pollution; water quality is declining."
    elif bod_value <= 10:
        status = "[UNSAFE] Unsafe"
        description = "Harmful to aquatic life; indicates significant pollution."
    elif bod_value <= 50:
        status = "[UNSAFE] Unsafe"
        description = "Unsuitable for most uses; indicates sewage contamination."
    else:
        status = "[UNSAFE] Unsafe"
        description = "Extremely polluted - industrial or raw sewage."
    
    return status, description

#Prediction
def predict_using_max():
    # List of feature names
    feature_names = X.columns.tolist()

    user_values = {}

    print("Enter the following feature values:\n")

    # Take user input for each feature
    for feature in feature_names:
        val = float(input(f"{feature}: "))
        user_values[feature] = [val]

    # Convert to DataFrame
    sample_df = pd.DataFrame(user_values)

    # Scale + predict
    sample_scaled = standardscaler.transform(sample_df)
    predicted_bod = model_full.predict(sample_scaled)
    bod_value = predicted_bod[0]
    
    status, description = interpret_bod_level(bod_value)
    print(f"\nPredicted BOD_Max: {bod_value:.2f} mg/L")
    print(f"Status: {status}")
    print(f"Interpretation: {description}")

def predict_using_top5():
    # List of top 5 feature names
    feature_names = top_5_features

    user_values = {}

    print("Enter the following feature values:\n")

    # Take user input for each feature
    for feature in feature_names:
        val = float(input(f"{feature}: "))
        user_values[feature] = [val]

    # Convert to DataFrame
    sample_df = pd.DataFrame(user_values)

    # Scale + predict - use the top 5 scaler
    sample_scaled = standardscaler_top5.transform(sample_df)
    predicted_bod = model_top5.predict(sample_scaled)
    bod_value = predicted_bod[0]
    
    status, description = interpret_bod_level(bod_value)
    print(f"\nPredicted BOD_Max (Top 5 Model): {bod_value:.2f} mg/L")
    print(f"Status: {status}")
    print(f"Interpretation: {description}")

if __name__ == "__main__":
    print("\nChoose Prediction Model:")
    print("1. Full Model (17 Features)")
    print("2. Top 5 Features Model")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        predict_using_max()
    elif choice == '2':
        predict_using_top5()
    else:
        print("Invalid choice. Please enter 1 or 2.")

#Deploy the code
import pickle
with open('model_top5.pkl', 'wb') as file:
    pickle.dump(model_top5, file)

with open('model_full.pkl', 'wb') as file:

    pickle.dump(model_full, file)
