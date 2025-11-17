import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).with_name("cleaned_river_dataset_median.csv")
MODEL_FULL_PATH = Path(__file__).with_name("model_full.pkl")
MODEL_TOP5_PATH = Path(__file__).with_name("model_top5.pkl")
RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    required_columns = {"BOD_Max"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Dataset missing required column(s): {missing}")

    df = df.drop(["Station_Code", "Monitoring_Location", "State"], axis=1, errors="ignore")
    df = df.dropna().reset_index(drop=True)

    X = df.drop("BOD_Max", axis=1)
    y = df["BOD_Max"]
    return X, y


def interpret_bod_level(bod_value: float) -> Tuple[str, str, str]:
    if bod_value <= 3:
        return "SAFE", "This water is generally clean and supports aquatic life.", "success"
    if bod_value <= 5:
        return "CAUTION", "Water quality is declining; monitor closely.", "warning"
    if bod_value <= 10:
        return "UNSAFE", "Significant pollution; harmful for most aquatic life.", "error"
    if bod_value <= 50:
        return "UNSAFE", "Severe contamination; indicates sewage impact.", "error"
    return "UNSAFE", "Extremely polluted — likely industrial or raw sewage.", "error"


def load_or_train_model(
    path: Path, X_train_scaled: np.ndarray, y_train: pd.Series
) -> Tuple[RandomForestRegressor, bool]:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f), False

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=20,
        max_features="sqrt",
        min_samples_split=8,
        min_samples_leaf=1,
        bootstrap=False,
    )
    model.fit(X_train_scaled, y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model, True


def compute_metrics(model: RandomForestRegressor, X_scaled: np.ndarray, y_true: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_scaled)
    mse = mean_squared_error(y_true, preds)
    return {
        "r2": r2_score(y_true, preds),
        "mae": mean_absolute_error(y_true, preds),
        "mse": mse,
        "rmse": np.sqrt(mse),
    }


def build_input_form(feature_list: List[str], feature_stats: pd.DataFrame) -> Dict[str, float]:
    input_values: Dict[str, float] = {}
    columns = st.columns(2)
    for idx, feature in enumerate(feature_list):
        stats = feature_stats.loc[feature]
        min_val = float(stats["min"])
        max_val = float(stats["max"])
        lower_bound = min(min_val, 0.0)
        upper_bound = max(max_val, 0.0)
        default_val = 0.0
        step = max((max_val - min_val) / 200, 0.01)
        if lower_bound == upper_bound:
            lower_bound = upper_bound - 1.0
        column = columns[idx % 2]
        input_values[feature] = column.number_input(
            label=feature,
            value=default_val,
            min_value=lower_bound,
            max_value=upper_bound,
            step=step,
            format="%.4f",
        )
    return input_values


def main() -> None:
    st.set_page_config(page_title="River BOD Predictor", page_icon="🌊", layout="wide")
    st.title("🌊 River Water BOD Predictor")
    st.write(
        "Predict the Biological Oxygen Demand (BOD) to assess river water quality. "
        "Choose the full feature model or a lightweight top-5 feature model."
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Please add the CSV and reload.")
        return

    df_raw = load_dataset(str(DATA_PATH))
    try:
        X, y = preprocess_data(df_raw)
    except ValueError as err:
        st.error(str(err))
        return

    feature_columns = X.columns.tolist()
    feature_stats = X.describe().T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    model_full, retrained_full = load_or_train_model(MODEL_FULL_PATH, X_train_scaled, y_train)

    feature_importance_df = (
        pd.DataFrame(
            {"Feature": feature_columns, "Importance": model_full.feature_importances_}
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )
    top5_features = feature_importance_df["Feature"].head(5).tolist()

    scaler_top5 = StandardScaler()
    X_train_top5_scaled = scaler_top5.fit_transform(X_train[top5_features])
    X_test_top5_scaled = scaler_top5.transform(X_test[top5_features])

    model_top5, retrained_top5 = load_or_train_model(
        MODEL_TOP5_PATH, X_train_top5_scaled, y_train
    )

    metrics_full_test = compute_metrics(model_full, X_test_scaled, y_test)
    metrics_top5_test = compute_metrics(model_top5, X_test_top5_scaled, y_test)
    metrics_full_train = compute_metrics(model_full, X_train_scaled, y_train)
    metrics_top5_train = compute_metrics(model_top5, X_train_top5_scaled, y_train)

    if retrained_full or retrained_top5:
        st.info(
            "Models were retrained because pre-trained files were missing. "
            "Saved fresh versions to disk for future runs."
        )

    st.subheader("Model Performance Snapshot")
    tabs = st.tabs(["Test data", "Training data"])

    def render_metrics(columns_container, metrics_full, metrics_top5):
        with columns_container[0]:
            st.markdown("**Full Model (all features)**")
            st.metric("R²", f"{metrics_full['r2']:.3f}")
            st.metric("MAE", f"{metrics_full['mae']:.3f}")
            st.metric("MSE", f"{metrics_full['mse']:.3f}")
            st.metric("RMSE", f"{metrics_full['rmse']:.3f}")
        with columns_container[1]:
            st.markdown("**Top-5 Feature Model**")
            st.metric("R²", f"{metrics_top5['r2']:.3f}")
            st.metric("MAE", f"{metrics_top5['mae']:.3f}")
            st.metric("MSE", f"{metrics_top5['mse']:.3f}")
            st.metric("RMSE", f"{metrics_top5['rmse']:.3f}")

    with tabs[0]:
        render_metrics(st.columns(2), metrics_full_test, metrics_top5_test)
    with tabs[1]:
        render_metrics(st.columns(2), metrics_full_train, metrics_top5_train)

    with st.expander("Feature Importance (Full Model)"):
        st.dataframe(feature_importance_df, use_container_width=True)

    model_choice = st.radio(
        "Choose a prediction mode",
        (
            "Full model (all features)",
            "Top-5 feature model",
        ),
        index=0,
        horizontal=True,
    )

    if model_choice.startswith("Full"):
        active_features = feature_columns
        scaler = scaler_full
        active_model = model_full
        model_label = "Full Model"
    else:
        active_features = top5_features
        scaler = scaler_top5
        active_model = model_top5
        model_label = "Top-5 Model"

    st.subheader("Enter Feature Values")
    with st.form("prediction_form"):
        user_inputs = build_input_form(active_features, feature_stats)
        submitted = st.form_submit_button("Predict BOD")

    if submitted:
        sample_df = pd.DataFrame([user_inputs])
        sample_df = sample_df.reindex(columns=active_features)
        sample_scaled = scaler.transform(sample_df)
        predicted_bod = float(active_model.predict(sample_scaled)[0])
        status, description, alert_type = interpret_bod_level(predicted_bod)

        if alert_type == "success":
            st.success(f"{model_label} Prediction: {predicted_bod:.2f} mg/L ({status})")
        elif alert_type == "warning":
            st.warning(f"{model_label} Prediction: {predicted_bod:.2f} mg/L ({status})")
        else:
            st.error(f"{model_label} Prediction: {predicted_bod:.2f} mg/L ({status})")

        st.write(description)

    


if __name__ == "__main__":
    main()


