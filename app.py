from flask import Flask, request, jsonify
import pandas as pd
import joblib
from predict_damage_for_city import predict_damage_for_city
import predict_damage_with_model
from haversine_distance import haversine_distance

app = Flask(__name__)

# ====== Load Model and Scalers ======
model = joblib.load("rf_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ====== Load Data ======
df_asteroids = pd.read_csv("top_100_real_nasa_asteroids_with_coords.csv")
df_cities = pd.read_csv("worldcities.csv", usecols=["city","lat","lng","country"])

# Clean text columns
df_asteroids["Object"] = df_asteroids["Object"].astype(str).str.strip().str.lower()
df_cities["city"] = df_cities["city"].astype(str).str.strip().str.lower()

# Drop rows with missing coordinates
df_asteroids = df_asteroids.dropna(subset=['asteroid_lat','asteroid_lon'])
df_cities = df_cities.dropna(subset=['lat','lng'])


#   Home Route

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API is running. Try /predict or /predict_city using POST"
    })


#   API 1: Predict Damage
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        diameter_m = data.get("diameter_m", 0)

        # Case: "Earth completely destroyed"
        if diameter_m >= 12742000:
            return jsonify({
                "message": "Asteroid diameter exceeds Earth's diameter! Earth is completely destroyed!"
            })

        if "city" in data:
            city_name = data["city"].strip().lower()
            city_row = df_cities[df_cities["city"] == city_name]

            if city_row.empty:
                return jsonify({"error": f"City '{data['city']}' not found"}), 400
            city_row = city_row.iloc[0]

            if "asteroid" in data and data["asteroid"]:
                # Use traditional predict_damage_for_city function
                result = predict_damage_for_city(
                    asteroid=data["asteroid"],
                    city=data["city"],
                    df_asteroids=df_asteroids,
                    df_cities=df_cities,
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y
                )
            else:
                # Use direct asteroid data
                crater, blast = predict_damage_with_model.predict_damage_with_model(
                    diameter_m=data["diameter_m"],
                    velocity_kms=data["velocity_kms"],
                    lat=city_row["lat"],
                    lon=city_row["lng"],
                    delta_km=data.get("delta_km", 1000),
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y
                )

                distance = 0.0
                is_city_affected = distance < blast

                result = {
                    "crater_diam_km": round(crater, 2),
                    "blast_radius_km": round(blast, 2),
                    "distance_km": round(distance, 2),
                    "is_city_affected": is_city_affected
                }

            return jsonify(result)

        else:
            # Predict using coordinates directly
            crater, blast = predict_damage_with_model.predict_damage_with_model(
                diameter_m=data["diameter_m"],
                velocity_kms=data["velocity_kms"],
                lat=data.get("lat", 0.0),
                lon=data.get("lon", 0.0),
                delta_km=data.get("delta_km", 1000),
                model=model,
                scaler_X=scaler_X,
                scaler_y=scaler_y
            )
            return jsonify({
                "crater_diam_km": round(crater, 2),
                "blast_radius_km": round(blast, 2)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

#   API 2: Predict City Damage

@app.route("/predict_city", methods=["POST"])
def predict_city():
    try:
        data = request.json
        asteroid = data["asteroid"]
        city = data["city"]

        result = predict_damage_for_city(
            asteroid, city, df_asteroids, df_cities,
            model, scaler_X, scaler_y
        )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
