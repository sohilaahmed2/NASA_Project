from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import geojson

app = Flask(__name__)
CORS(app) 

# ======================================================
# === USGS Elevation API ===============================
# ======================================================
def get_elevation_usgs(lat, lon):
    try:
        url = f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=Meters&output=json"
        r = requests.get(url, timeout=6)
        data = r.json()
        elevation = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        return float(elevation)
    except Exception as e:
        print(f"[get_elevation_usgs] Error: {e}")
        return None

def is_water(lat, lon):
    elev = get_elevation_usgs(lat, lon)
    if elev is not None:
        return elev <= 0, "usgs_api", elev
    return False, "fallback_default_land", None

# ======================================================
# === Damage Calculations ==============================
# ======================================================
def calculate_energy(diameter_m, velocity_kms, density=3000):
    radius_m = diameter_m / 2.0
    volume_m3 = (4.0 / 3.0) * math.pi * (radius_m ** 3)
    mass_kg = volume_m3 * density
    velocity_ms = velocity_kms * 1000.0
    energy_joules = 0.5 * mass_kg * (velocity_ms ** 2)
    return energy_joules

def calculate_crater_diameter(diameter_m, velocity_kms):
    return diameter_m * (velocity_kms ** 0.44)

def calculate_blast_radius(energy_joules):
    return (energy_joules ** (1.0 / 3.0)) / 1000.0  # km

def calculate_earthquake_magnitude(energy_joules):
    if energy_joules <= 0:
        return 0.0
    return (2.0 / 3.0) * (math.log10(energy_joules) - 4.8)

# ======================================================
# === GeoJSON Generator ================================
# ======================================================
def make_geojson(lat, lon, blast_radius_km, crater_diam_m):
    features = [
        geojson.Feature(
            geometry=geojson.Point((lon, lat)),
            properties={"role": "impact_point"}
        ),
        geojson.Feature(
            geometry=geojson.Point((lon, lat)),
            properties={"role": "blast_zone_center", "radius_km": blast_radius_km}
        ),
        geojson.Feature(
            geometry=geojson.Point((lon, lat)),
            properties={"role": "crater_center", "radius_m": crater_diam_m}
        )
    ]
    return geojson.FeatureCollection(features)

# ======================================================
# === Haversine Distance ===============================
# ======================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ======================================================
# === Volcanic Impact =================================
# ======================================================
def is_volcanic_area(lat, lon, energy_joules):
    volcanoes = [
        {"name": "Mount St. Helens", "lat": 46.2, "lon": -122.18, "radius_km": 50},
        {"name": "Kilauea", "lat": 19.4, "lon": -155.3, "radius_km": 50},
        {"name": "Mount Rainier", "lat": 46.85, "lon": -121.75, "radius_km": 50}
    ]
    for v in volcanoes:
        dist = haversine_distance(lat, lon, v["lat"], v["lon"])
        if dist <= v["radius_km"]:
            if energy_joules > 1e19:
                impact_level = "high"
            elif energy_joules > 1e17:
                impact_level = "medium"
            else:
                impact_level = "low"
            return True, v["name"], impact_level
    return False, None, None

# ======================================================
# === Flask Routes =====================================
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🌍 Asteroid Impact API is running (volcanic impact only, no tsunami)!",
        "usage": {
            "endpoint": "/impact",
            "method": "POST",
            "example_payload": {
                "diameter_m": 500,
                "velocity_kms": 20,
                "lat": 36.0,
                "lon": -120.0
            }
        }
    })

@app.route("/impact", methods=["POST"])
def impact():
    data = request.get_json(force=True)

    diameter_m = float(data.get("diameter_m", 100.0))
    velocity_kms = float(data.get("velocity_kms", 20.0))
    lat = float(data.get("lat", 0.0))
    lon = float(data.get("lon", 0.0))

    EARTH_DIAMETER_M = 12742000
    catastrophic_flag = False
    catastrophic_message = None
    if diameter_m >= EARTH_DIAMETER_M:
        catastrophic_flag = True
        catastrophic_message = "Total destruction!"
    # calculations
    energy = calculate_energy(diameter_m, velocity_kms)
    crater_diam_m = calculate_crater_diameter(diameter_m, velocity_kms)
    crater_diam_km = crater_diam_m / 1000.0
    blast_radius_km = calculate_blast_radius(energy)
    magnitude = calculate_earthquake_magnitude(energy)

    # water check (keep for reference, but no tsunami calculation)
    water_bool, water_source, elevation_m = is_water(lat, lon)

    # volcanic check
    volcanic_bool, volcano_name, impact_level = is_volcanic_area(lat, lon, energy)

    # GeoJSON
    geojson_data = make_geojson(lat, lon, blast_radius_km, crater_diam_m)

    return jsonify({
        "input": {
            "diameter_m": diameter_m,
            "velocity_kms": velocity_kms,
            "lat": lat,
            "lon": lon
        },
        "location": {
            "is_water": water_bool,
            "is_water_source": water_source,
            "elevation_m": elevation_m
        },
        "results": {
            "energy_joules": energy,
            "crater_diameter_m": crater_diam_m,
            "crater_diameter_km": crater_diam_km,
            "blast_radius_km": blast_radius_km,
            "earthquake_magnitude": magnitude
        },
        "volcanic_impact": {
            "is_affected": volcanic_bool,
            "volcano_name": volcano_name,
            "impact_level": impact_level
        },
        "catastrophic_impact": {
           "catastrophic_destruction": catastrophic_flag,
           "catastrophic_message": catastrophic_message
        },
        "geojson": geojson_data
    })

# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
