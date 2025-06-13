import os
import joblib 
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from flask_cors import CORS

# --- FUNGSI PEMUATAN ASET ML ---
def load_model_assets(model_dir='models_ml'):
    try:
        print("--- Memuat Aset Machine Learning ---")
        model_path = os.path.join(model_dir, "model_best_tf.h5")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model berhasil dimuat dari '{model_path}'")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler berhasil dimuat dari '{scaler_path}'")
        label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        print(f"✅ Label Encoder berhasil dimuat dari '{label_encoder_path}'")
        print("--- Semua aset berhasil dimuat. ---")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"❌ Gagal memuat aset. Error: {e}")
        return None, None, None

# --- INISIALISASI APLIKASI DAN ASET ML ---
load_dotenv()
app = Flask(__name__)
weather_model, scaler, label_encoder = load_model_assets()

# --- KONFIGURASI APLIKASI FLASK ---
CORS(app)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "super-secret-key-for-dev")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'weather_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- MODEL DATABASE (TABEL) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade="all, delete-orphan")

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temp_min_c = db.Column(db.Float, nullable=False)
    temp_max_c = db.Column(db.Float, nullable=False)
    temp_avg_c = db.Column(db.Float, nullable=False)
    humidity_avg_percent = db.Column(db.Float, nullable=False)
    precip_mm = db.Column(db.Float, nullable=False)
    sunshine_duration_hours = db.Column(db.Float, nullable=False)
    wind_speed_max_ms = db.Column(db.Float, nullable=False)
    wind_dir_max_deg = db.Column(db.Float, nullable=False)
    wind_speed_avg_ms = db.Column(db.Float, nullable=False)
    predicted_weather = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# --- FUNGSI HELPER UNTUK PREDIKSI ---
def run_prediction_logic(data):
    """Fungsi terpusat untuk menjalankan logika prediksi."""
    expected_features = [
        'temp_min_c', 'temp_max_c', 'temp_avg_c', 'humidity_avg_percent', 
        'precip_mm', 'sunshine_duration_hours', 'wind_speed_max_ms', 
        'wind_dir_max_deg', 'wind_speed_avg_ms'
    ]
    if not data or not all(key in data for key in expected_features):
        raise ValueError("Data input tidak lengkap")
    
    feature_values = [float(data[key]) for key in expected_features]
    features = np.array([feature_values])
    scaled_features = scaler.transform(features)
    probabilities = weather_model.predict(scaled_features)
    prediction_code = np.argmax(probabilities, axis=1)[0]
    predicted_weather_label = label_encoder.inverse_transform([int(prediction_code)])[0]
    return str(predicted_weather_label)

# --- RUTE API (API ENDPOINTS) ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Username dan password dibutuhkan"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username sudah terdaftar"}), 409
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User berhasil dibuat"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=str(user.id))
        return jsonify(access_token=access_token), 200
    return jsonify({"error": "Username atau password salah"}), 401

# PERBARUAN: Endpoint ini untuk prediksi utama DAN MENYIMPAN ke riwayat
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_and_save():
    if not all([weather_model, scaler, label_encoder]):
        return jsonify({"error": "Aset machine learning tidak tersedia."}), 503
    
    data = request.get_json()
    try:
        predicted_weather = run_prediction_logic(data)
        
        current_user_id = get_jwt_identity()
        new_history = PredictionHistory(
            **data, 
            predicted_weather=predicted_weather, 
            user_id=current_user_id
        )
        db.session.add(new_history)
        db.session.commit()
        
        return jsonify({"predicted_weather": predicted_weather}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

# BARU: Endpoint ini HANYA untuk prediksi (TANPA MENYIMPAN)
@app.route('/predict-no-save', methods=['POST'])
@jwt_required()
def predict_no_save():
    if not all([weather_model, scaler, label_encoder]):
        return jsonify({"error": "Aset machine learning tidak tersedia."}), 503
    
    data = request.get_json()
    try:
        predicted_weather = run_prediction_logic(data)
        return jsonify({"predicted_weather": predicted_weather}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    user_history = PredictionHistory.query.filter_by(user_id=current_user_id).order_by(PredictionHistory.timestamp.desc()).limit(20).all()
    history_list = [
        {
            "id": record.id,
            "inputs": {
                "temp_min_c": record.temp_min_c, "temp_max_c": record.temp_max_c,
                "temp_avg_c": record.temp_avg_c, "humidity_avg_percent": record.humidity_avg_percent,
                "precip_mm": record.precip_mm, "sunshine_duration_hours": record.sunshine_duration_hours,
                "wind_speed_max_ms": record.wind_speed_max_ms, "wind_dir_max_deg": record.wind_dir_max_deg,
                "wind_speed_avg_ms": record.wind_speed_avg_ms,
            },
            "prediction": record.predicted_weather,
            "timestamp": record.timestamp.isoformat()
        } for record in user_history
    ]
    return jsonify(history_list), 200

@app.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if user:
        return jsonify(id=user.id, username=user.username), 200
    return jsonify({"error": "User not found"}), 404


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)