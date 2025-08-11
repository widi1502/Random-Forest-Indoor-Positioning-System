import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="📶 Prediksi Lokasi WiFi", layout="centered")
st.title("📍 Prediksi Lokasi Berdasarkan RSSI")

# Load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset untuk ambil nama kolom
df = pd.read_csv("dataset_model.csv")
all_features = df.drop(columns='spot').columns.tolist()   # 78 kolom total
feature_columns = all_features[:5]  # hanya pakai 5 kolom untuk input manual

# Mode input
mode = st.selectbox("🔧 Pilih Mode Input RSSI", ("Pilih...", "Manual", "Upload File CSV", "WiFi Snapshot (Otomatis)"))

# === MODE MANUAL (input 5 kolom, isi lainnya -100) ===
if mode == "Manual":
    st.subheader("🖊️ Input Manual RSSI")
    st.info("Masukkan nilai RSSI untuk 5 AP. Sisanya akan diisi otomatis dengan -100.")

    manual_input = []
    with st.form("manual_form"):
        for col in feature_columns:
            val = st.number_input(f"{col}", min_value=-100, max_value=0, value=-70)
            manual_input.append(val)
        submitted = st.form_submit_button("Prediksi Lokasi")

    if submitted:
        # Buat input penuh untuk semua fitur
        full_input = pd.DataFrame([[-100]*len(all_features)], columns=all_features)
        for i, col in enumerate(feature_columns):
            full_input[col] = manual_input[i]

        prediction = model.predict(full_input)[0]
        st.success(f"📌 Lokasi Diprediksi: **{prediction}**")

# === MODE UPLOAD CSV ===
elif mode == "Upload File CSV":
    st.subheader("📄 Upload File CSV RSSI")
    st.markdown("""
                📌 **Petunjuk Format CSV:**
- File harus berupa `.csv`
- Berisi **kolom RSSI (AP1, AP2, ..., AP78)** sesuai model
- **Tanpa kolom `spot`**
- Nilai RSSI: angka antara **-100 hingga 0**

Contoh baris:
AP1,AP2,AP3,...,AP78
-65,-70,-85,...,-60
-62,-68,-81,...,-57""")
    uploaded_file = st.file_uploader("Pilih file .csv", type="csv")

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if set(uploaded_df.columns) != set(all_features):
                st.error("❌ Kolom pada CSV tidak sesuai dengan model.")
            else:
                st.dataframe(uploaded_df.head())
                if st.button("Prediksi Semua Baris"):
                    preds = model.predict(uploaded_df)
                    uploaded_df['Predicted Spot'] = preds
                    st.success("✅ Prediksi berhasil!")
                    st.dataframe(uploaded_df)

                    csv_out = uploaded_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Unduh Hasil Prediksi", data=csv_out, file_name="predicted_output.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# === MODE SNAPSHOT (Simulasi acak) ===
elif mode == "WiFi Snapshot (Otomatis)":
    st.subheader("📶 WiFi Snapshot (Simulasi)")
    st.warning("🔧 Fitur ini masih simulasi. Menghasilkan RSSI acak.")
    
    if st.button("Ambil Snapshot WiFi (Simulasi)"):
        rssi_values = np.random.randint(-90, -40, size=len(all_features))
        snapshot_df = pd.DataFrame([rssi_values], columns=all_features)
        st.dataframe(snapshot_df)

        pred = model.predict(snapshot_df)[0]
        st.success(f"📌 Lokasi Diprediksi dari Snapshot: **{pred}**")
        
# ===========================================
# Cara Menjalankan Aplikasi Prediksi RSSI:
# 1. Buka Terminal / Command Prompt
# 2. Masuk ke folder tempat file ini berada
# 3. Jalankan perintah:
#    streamlit run prototype_rf.py
# 4. Buka browser dan akses:
#    http://localhost:8501
# ===========================================