import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from flask_cors import CORS
import pickle # Menggunakan pickle untuk memuat file .save
import numpy as np
from tensorflow.keras.models import load_model # Untuk memuat model .h5
from datetime import datetime
from dotenv import load_dotenv

# --- INISIALISASI & KONFIGURASI APLIKASI ---
load_dotenv() # Memuat environment variables dari file .env

app = Flask(__name__)
CORS(app) # Mengizinkan akses API dari domain lain (misal: frontend React)

# Konfigurasi dari environment variables atau nilai default
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "fallback-secret-key-jika-env-tidak-ada")
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'weather_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi ekstensi
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)


# --- MEMUAT MODEL MACHINE LEARNING & ASET LAINNYA ---
# Model, scaler, dan encoder dimuat sekali saat aplikasi start untuk efisiensi
weather_model = None
scaler = None
label_encoder = None

try:
    model_path = os.path.join('models_ml', 'model_tf_weather.h5')
    scaler_path = os.path.join('models_ml', 'scaler.save')
    encoder_path = os.path.join('models_ml', 'label_encoder.save')
    
    if not os.path.exists(model_path):
        print(f"ERROR: File model tidak ditemukan di {model_path}")
    elif not os.path.exists(scaler_path):
        print(f"ERROR: File scaler tidak ditemukan di {scaler_path}")
    elif not os.path.exists(encoder_path):
        print(f"ERROR: File label encoder tidak ditemukan di {encoder_path}")
    else:
        # Memuat model Keras .h5
        weather_model = load_model(model_path)
        
        # Memuat scaler dan encoder yang disimpan dengan pickle
        with open(scaler_path, 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        with open(encoder_path, 'rb') as f_encoder:
            label_encoder = pickle.load(f_encoder)

        print("Model Keras, Scaler, dan Label Encoder berhasil dimuat.")

except Exception as e:
    print(f"ERROR saat memuat aset ML: {e}")
    # Biarkan weather_model, scaler, label_encoder tetap None


# --- MODEL DATABASE (TABEL) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # Relasi: Satu User bisa memiliki banyak PredictionHistory
    # cascade="all, delete-orphan" berarti jika User dihapus, semua historynya juga terhapus
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade="all, delete-orphan")

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    pressure = db.Column(db.Float, nullable=False)
    predicted_weather = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # Foreign Key yang menghubungkan ke tabel User
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# --- RUTE API (API ENDPOINTS) ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username dan password dibutuhkan"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username sudah terdaftar"}), 409 # Conflict

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User berhasil dibuat"}), 201 # Created
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Gagal menyimpan user: {str(e)}"}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username dan password dibutuhkan"}), 400

    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token), 200
    
    return jsonify({"error": "Username atau password salah"}), 401 # Unauthorized


@app.route('/predict', methods=['POST'])
@jwt_required() # Endpoint ini diamankan, butuh token untuk akses
def predict():
    # Periksa apakah semua aset ML sudah termuat
    if not all([weather_model, scaler, label_encoder]):
        print("Peringatan: Aset ML tidak termuat saat mencoba prediksi.")
        return jsonify({"error": "Aset machine learning tidak tersedia atau gagal dimuat"}), 503 # Service Unavailable

    # 1. Dapatkan data input dari request
    data = request.get_json()
    if not data or not all(key in data for key in ['temperature', 'humidity', 'pressure']):
        return jsonify({"error": "Data input tidak lengkap (membutuhkan temperature, humidity, pressure)"}), 400

    try:
        current_user_id = get_jwt_identity() # Dapatkan ID user dari token JWT
        
        # Validasi tipe data input
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])

        # 2. Siapkan data menjadi array numpy 2D (model Keras mengharapkan input batch)
        features = np.array([[temp, humidity, pressure]])

        # 3. Scale data menggunakan scaler yang telah dimuat
        scaled_features = scaler.transform(features)

        # 4. Lakukan prediksi dengan model Keras
        # Model Keras .predict() menghasilkan probabilitas untuk setiap kelas
        probabilities = weather_model.predict(scaled_features)
        
        # 5. Ambil indeks dari probabilitas tertinggi (ini adalah prediksi dalam bentuk angka)
        prediction_code = np.argmax(probabilities, axis=1)[0] 
        
        # 6. Ubah kode prediksi (angka) menjadi label teks menggunakan encoder
        # Pastikan prediction_code adalah tipe yang diharapkan oleh inverse_transform (biasanya list atau array)
        predicted_weather_label = label_encoder.inverse_transform([int(prediction_code)])[0]

        # 7. Simpan hasil ke database
        new_history = PredictionHistory(
            temperature=temp,
            humidity=humidity,
            pressure=pressure,
            predicted_weather=str(predicted_weather_label), # Pastikan label adalah string
            user_id=current_user_id
        )
        db.session.add(new_history)
        db.session.commit()

        # 8. Kembalikan hasil prediksi ke frontend
        return jsonify({
            "predicted_weather": str(predicted_weather_label) 
        }), 200

    except ValueError:
        return jsonify({"error": "Input temperature, humidity, dan pressure harus berupa angka"}), 400
    except Exception as e:
        db.session.rollback() # Rollback jika ada error saat transaksi DB
        print(f"Error di endpoint /predict: {str(e)}")
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    # Ambil 20 histori terbaru untuk user yang sedang login
    user_history_query = PredictionHistory.query.filter_by(user_id=current_user_id).order_by(PredictionHistory.timestamp.desc()).limit(20)
    user_history = user_history_query.all()
    
    history_list = [
        {
            "id": record.id,
            "inputs": {
                "temperature": record.temperature,
                "humidity": record.humidity,
                "pressure": record.pressure
            },
            "prediction": record.predicted_weather,
            "timestamp": record.timestamp.isoformat() # Format ISO agar mudah diproses di frontend
        } for record in user_history
    ]
        
    return jsonify(history_list), 200


# --- MENJALANKAN APLIKASI ---
if __name__ == '__main__':
    # Membuat direktori instance jika belum ada (penting untuk SQLite)
    instance_path = os.path.join(basedir, 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        print(f"Direktori 'instance' dibuat di {instance_path}")

    with app.app_context():
        db.create_all() # Membuat tabel database jika belum ada
        print("Tabel database diperiksa/dibuat.")
    
    print("Menjalankan server Flask...")
    app.run(debug=True, port=5000)