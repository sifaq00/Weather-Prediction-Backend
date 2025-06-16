import tensorflow as tf
import os
import shutil

print("Memulai proses konversi model dengan metode 'export' (Keras 3)...")

# Tentukan path file dan direktori
output_dir = 'models_ml'
h5_model_path = os.path.join(output_dir, 'model_best_tf.h5')
saved_model_dir = os.path.join(output_dir, 'saved_model_dir')
tflite_model_path = os.path.join(output_dir, 'model_best_tf.tflite')

# Hapus direktori saved_model lama jika ada, untuk memastikan kebersihan
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)
    print(f"-> Menghapus direktori '{saved_model_dir}' yang lama.")

# LANGKAH 1: Muat model .h5 yang sudah ada
try:
    model = tf.keras.models.load_model(h5_model_path, compile=False) 
    print(f"-> Model '{h5_model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL: Tidak dapat memuat model .h5.")
    print(f"Error: {e}")
    exit()

# LANGKAH 2: Ekspor ke format 'SavedModel' menggunakan metode baru .export()
try:
    print(f"-> Mengekspor model ke format SavedModel di '{saved_model_dir}'...")
    # DIUBAH: Gunakan model.export() sesuai anjuran error Keras 3
    model.export(saved_model_dir) 
    print("-> Berhasil mengekspor ke format SavedModel.")
except Exception as e:
    print(f"GAGAL: Tidak dapat mengekspor ke format SavedModel.")
    print(f"Error: {e}")
    exit()
        
# LANGKAH 3: Konversi ke TFLite dari direktori SavedModel
try:
    print(f"-> Mengonversi model dari '{saved_model_dir}' ke TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
except Exception as e:
    print(f"GAGAL: Terjadi error saat proses konversi TFLite.")
    print(f"Error: {e}")
    exit()

# LANGKAH 4: Simpan file .tflite yang sudah jadi
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"\nSUKSES! Model berhasil dikonversi dan disimpan di: {tflite_model_path}")