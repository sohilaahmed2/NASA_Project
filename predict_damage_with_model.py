import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def predict_damage_with_model(diameter_m: float, velocity_kms: float, lat: float = 0.0, lon: float = 0.0, delta_km: float = 1000.0, model=None, scaler_X: MinMaxScaler = None, scaler_y: MinMaxScaler = None) -> tuple:
    """
    Predicts damage (crater diameter and blast radius) using the trained RandomForest model instead of physical equations.
    
    Args:
        diameter_m (float): Asteroid diameter in meters (>= 1.0).
        velocity_kms (float): Asteroid velocity in km/s (>= 0.1).
        lat (float, optional): Asteroid latitude (default: 0.0).
        lon (float, optional): Asteroid longitude (default: 0.0).
        delta_km (float, optional): Close approach distance in km (default: 1000.0).
        model: Trained RandomForest model (loaded from rf_model.pkl).
        scaler_X (MinMaxScaler): Scaler for input features.
        scaler_y (MinMaxScaler): Scaler for output predictions.
    
    Returns:
        tuple: (crater_diam_km, blast_radius_km) in kilometers.
    
    Raises:
        ValueError: If inputs are invalid or model/scalers not provided.
        Exception: For other prediction errors.
    """
    try:
        # Validate inputs
        if diameter_m < 1.0 or velocity_kms < 0.1:
            raise ValueError(f"المدخلات غير صالحة: القطر يجب أن يكون >= 1 متر، والسرعة >= 0.1 كم/ث")
        if model is None or scaler_X is None or scaler_y is None:
            raise ValueError("النموذج أو الموسعات غير متوفرة")
        
        # Prepare features (same as used in training)
        features = np.array([[diameter_m, velocity_kms, lat, lon, delta_km]])
        
        # Scale features, predict, and inverse scale
        features_scaled = scaler_X.transform(features)
        pred_scaled = model.predict(features_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)[0]
        
        # Return as tuple to match calculate_damage output
        return float(pred[0]), float(pred[1])
    except ValueError as ve:
        print(f"خطأ في المدخلات: {str(ve)}")
        return np.nan, np.nan
    except Exception as e:
        print(f"خطأ في التنبؤ: diameter={diameter_m}, velocity={velocity_kms}, error={str(e)}")
        return np.nan, np.nan