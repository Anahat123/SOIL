import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from data.dataset import get_dummy_data

def run_baselines():
    """
    Runs classical ML baselines independently for Moisture (Regression) and Texture (Classification)
    using Random Forest models. This is used to benchmark the Deep Learning Multi-Task models.
    """
    print("Generating synthetic 1D feature data...")
    # Baselines typically use 1D (flattened or spectral) features
    X_train, y_m_train, y_t_train = get_dummy_data(800, mode='1d')
    X_test, y_m_test, y_t_test = get_dummy_data(200, mode='1d')
    
    print("\n--- Task 1: Soil Moisture Prediction (Regression) ---")
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_m_train)
    
    preds_m = rf_regressor.predict(X_test)
    rmse = mean_squared_error(y_m_test, preds_m, squared=False)
    r2 = r2_score(y_m_test, preds_m)
    print(f"Random Forest Regressor -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    print("\n--- Task 2: Soil Texture Classification ---")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_t_train)
    
    preds_t = rf_classifier.predict(X_test)
    acc = accuracy_score(y_t_test, preds_t)
    f1 = f1_score(y_t_test, preds_t, average='macro')
    print(f"Random Forest Classifier -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")

if __name__ == '__main__':
    run_baselines()
