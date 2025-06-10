import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def train_and_save_model():
    """
    Fungsi ini akan melakukan semua langkah dari notebook:
    1. Memuat data.
    2. Melakukan pra-pemrosesan data.
    3. Melatih model MLP TensorFlow.
    4. Menyimpan model, scaler, dan label encoder.
    """
    print("--- 1. Memuat Dataset ---")
    try:
        url = 'https://raw.githubusercontent.com/melshitaardia/Weather-Prediction/main/datacuaca_BMKG.csv'
        df = pd.read_csv(url)
        print("Dataset berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat dataset. Error: {e}")
        return

    print("\n--- 2. Pra-pemrosesan Data ---")
    
    # Buat label cuaca berdasarkan curah hujan
    def generate_label(precip):
        if precip <= 0.2:
            return "cerah"
        elif precip <= 1.0:
            return "mendung"
        else:
            return "hujan"
    df['cuaca'] = df['precip_mm'].apply(generate_label)
    
    # Hapus kolom yang tidak diperlukan
    df_cleaned = df.drop(columns=['tanggal', 'source_file', 'wind_dir_common_deg'])
    
    # Label Encoding untuk target
    label_encoder = LabelEncoder()
    df_cleaned['cuaca_encoded'] = label_encoder.fit_transform(df_cleaned['cuaca'])
    
    # Pisahkan fitur (X) dan target (y)
    X = df_cleaned.drop(columns=['cuaca', 'cuaca_encoded'])
    y = df_cleaned['cuaca_encoded']
    
    # Simpan nama kolom fitur untuk backend
    feature_columns = X.columns.tolist()
    print(f"Fitur yang digunakan untuk training ({len(feature_columns)} fitur): {feature_columns}")
    
    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Pembagian data (train, validation, test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Balancing data training dengan SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Pra-pemrosesan data selesai.")

    print("\n--- 3. Membangun dan Melatih Model MLP TensorFlow ---")
    
    # Definisikan arsitektur model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_smote.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax') 
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping untuk mencegah overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Latih model
    history = model.fit(
        X_train_smote, y_train_smote,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1 
    )
    print("Pelatihan model selesai.")

    print("\n--- 4. Menyimpan Aset Model ---")
    
    # Buat folder 'models_ml' jika belum ada
    output_dir = 'models_ml'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' berhasil dibuat.")
    
    try:
        # Simpan model TensorFlow
        model.save(os.path.join(output_dir, "model_best_tf.h5"))
        print(f"✅ Model berhasil disimpan di '{os.path.join(output_dir, 'model_best_tf.h5')}'")
        
        # Simpan scaler
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        print(f"✅ Scaler berhasil disimpan di '{os.path.join(output_dir, 'scaler.pkl')}'")
        
        # Simpan label encoder
        joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
        print(f"✅ Label Encoder berhasil disimpan di '{os.path.join(output_dir, 'label_encoder.pkl')}'")
    
    except Exception as e:
        print(f"Gagal menyimpan aset. Error: {e}")

# Entry point untuk menjalankan skrip
if __name__ == '__main__':
    train_and_save_model()