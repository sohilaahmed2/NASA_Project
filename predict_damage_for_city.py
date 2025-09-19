import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from haversine_distance import haversine_distance  # استخدم الملف الموجود عندك

def normalize_lon(lon):
    # يحول أي قيمة longitude لتكون بين -180 و 180
    return ((lon + 180) % 360) - 180

def validate_coords(lat, lon):
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError(f"الإحداثيات غير صالحة: lat={lat}, lon={lon}")

def predict_damage_for_city(
    asteroid: str,
    city: str,
    df_asteroids: pd.DataFrame,
    df_cities: pd.DataFrame,
    model,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler
) -> dict:
    try:
        # تنظيف النصوص
        asteroid = asteroid.strip().lower()
        city = city.strip().lower()

        # تنظيف الأعمدة
        df_asteroids["Object"] = df_asteroids["Object"].astype(str).str.strip().str.lower()
        df_cities["city"] = df_cities["city"].astype(str).str.strip().str.lower()

        # حذف الصفوف الغير صالحة في الإحداثيات
        df_asteroids = df_asteroids.dropna(subset=['asteroid_lat','asteroid_lon'])
        df_cities = df_cities.dropna(subset=['lat','lng'])

        # اختيار الصفوف
        asteroid_row = df_asteroids[df_asteroids['Object'] == asteroid]
        city_row = df_cities[df_cities['city'] == city]

        if asteroid_row.empty:
            raise ValueError(f"الكويكب '{asteroid}' غير موجود في البيانات")
        if city_row.empty:
            raise ValueError(f"المدينة '{city}' غير موجودة في البيانات")

        asteroid_row = asteroid_row.iloc[0]
        city_row = city_row.iloc[0]

        # ===== تصحيح longitude =====
        asteroid_row['asteroid_lon'] = normalize_lon(asteroid_row['asteroid_lon'])
        city_row['lng'] = normalize_lon(city_row['lng'])

        # تحقق من الإحداثيات بعد التصحيح
        validate_coords(asteroid_row['asteroid_lat'], asteroid_row['asteroid_lon'])
        validate_coords(city_row['lat'], city_row['lng'])

        # تجهيز الميزات للتنبؤ
        features = np.array([[
            asteroid_row['diameter_m'],
            asteroid_row['V relative(km/s)'],
            asteroid_row['asteroid_lat'],
            asteroid_row['asteroid_lon'],
            asteroid_row['closest_delta_km']
        ]])

        features_scaled = scaler_X.transform(features)
        pred_scaled = model.predict(features_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)[0]

        # حساب المسافة باستخدام haversine_distance من الملف المحلي
        distance = haversine_distance(
            asteroid_row['asteroid_lat'], asteroid_row['asteroid_lon'],
            city_row['lat'], city_row['lng']
        )

        return {
            'crater_diam_km': float(pred[0]),
            'blast_radius_km': float(pred[1]),
            'distance_km': float(distance),
            'is_city_affected': bool(distance < pred[1])
        }

    except Exception as e:
        raise Exception(f"خطأ في التنبؤ لـ {asteroid}, {city}: {str(e)}")
