"""
SERVER AI: Smart Heart Rate Health Monitor
Versi: HYBRID (Machine Learning + Rule-Based) + Real-time Dashboard
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
from datetime import datetime
import json
import os
import joblib
import queue
import threading

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ==========================================
# ⭐ FIX: CUSTOM JSON ENCODER UNTUK NUMPY TYPES
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

# === KONFIGURASI ===
WINDOW_SIZE = 10
LOG_DIR = "health_logs"
MODEL_DIR = "ml_models"

for folder in [LOG_DIR, MODEL_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

device_data = {}
device_models = {}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_hrv_features(bpm_list):
    if len(bpm_list) < 3:
        return None
    bpm_array = np.array(bpm_list)
    mean_bpm = float(np.mean(bpm_array))
    sdnn = float(np.std(bpm_array))
    successive_diffs = np.diff(bpm_array)
    rmssd = float(np.sqrt(np.mean(successive_diffs**2))) if len(successive_diffs) > 0 else 0.0
    return {
        'mean_bpm': mean_bpm, 'sdnn': sdnn, 'rmssd': rmssd,
        'min_bpm': float(np.min(bpm_array)), 'max_bpm': float(np.max(bpm_array))
    }

def train_activity_classifier(motion_history, bpm_history):
    if len(motion_history) < 30:
        return None, "Data belum cukup"
    X_train, y_train = [], []
    for i in range(len(motion_history) - WINDOW_SIZE + 1):
        window_motion = motion_history[i:i+WINDOW_SIZE]
        window_bpm = bpm_history[i:i+WINDOW_SIZE]
        mean_motion = float(np.mean(window_motion))
        std_motion = float(np.std(window_motion))
        hrv = calculate_hrv_features(window_bpm)
        if hrv:
            X_train.append([mean_motion, std_motion, hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd']])
            if mean_motion < 15: y_train.append(0)
            elif mean_motion < 40: y_train.append(1)
            else: y_train.append(2)
    if len(X_train) < 10: return None, "Data training tidak cukup"
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf, f"Trained with {len(X_train)} samples, acc: {clf.score(X_train, y_train):.2%}"

def train_health_clustering(bpm_history, motion_history):
    if len(bpm_history) < 30: return None, None, "Data belum cukup"
    X_train = []
    for i in range(len(bpm_history) - WINDOW_SIZE + 1):
        window_bpm = bpm_history[i:i+WINDOW_SIZE]
        window_motion = motion_history[i:i+WINDOW_SIZE]
        hrv = calculate_hrv_features(window_bpm)
        if hrv:
            X_train.append([hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd'], float(np.mean(window_motion))])
    if len(X_train) < 10: return None, None, "Data tidak cukup"
    X_train = np.array(X_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_scores = [(100 - c[0]) + (c[1] * 10) for c in centroids_original]
    sorted_indices = np.argsort(cluster_scores)[::-1]
    cluster_labels = {int(sorted_indices[0]): 'EXCELLENT', int(sorted_indices[1]): 'GOOD', int(sorted_indices[2]): 'FAIR'}
    return kmeans, scaler, {'cluster_labels': cluster_labels, 'centroids': centroids_original.tolist(), 'scores': [float(cluster_scores[i]) for i in sorted_indices]}

def train_novelty_detector(bpm_history, motion_history):
    if len(bpm_history) < 25: return None, "Data baseline belum cukup"
    X_train = []
    for i in range(len(bpm_history) - WINDOW_SIZE + 1):
        window_bpm = bpm_history[i:i+WINDOW_SIZE]
        window_motion = motion_history[i:i+WINDOW_SIZE]
        hrv = calculate_hrv_features(window_bpm)
        if hrv:
            X_train.append([hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd'], float(np.mean(window_motion))])
    if len(X_train) < 10: return None, "Data tidak cukup"
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    iso_forest.fit(X_train)
    return iso_forest, f"Trained with {len(X_train)} baseline samples"

def calculate_health_score(mean_bpm, sdnn, context):
    if context == "REST":
        if mean_bpm < 60: bpm_score = 95
        elif mean_bpm < 70: bpm_score = 85
        elif mean_bpm < 80: bpm_score = 70
        elif mean_bpm < 90: bpm_score = 55
        else: bpm_score = 40
    else:
        if mean_bpm < 100: bpm_score = 80
        elif mean_bpm < 120: bpm_score = 65
        else: bpm_score = 50
    
    hrv_score = 85 if sdnn > 5 else (70 if sdnn > 3 else 55)
    health_score = int(bpm_score * 0.6 + hrv_score * 0.4)
    status = "EXCELLENT" if health_score >= 85 else ("GOOD" if health_score >= 70 else ("FAIR" if health_score >= 55 else "NEEDS ATTENTION"))
    return health_score, status

def generate_mcu_report(device_id, bpm_history, motion_history):
    """
    Generate laporan Medical Check-Up (MCU) berdasarkan data kalibrasi.
    Dipanggil sekali saat model selesai training.
    """
    import statistics
    
    # Ambil data kalibrasi (30 data terakhir)
    calibration_bpm = bpm_history[-30:]
    calibration_motion = motion_history[-30:]
    
    # === ANALISIS HEART RATE ===
    avg_bpm = float(np.mean(calibration_bpm))
    min_bpm = float(np.min(calibration_bpm))
    max_bpm = float(np.max(calibration_bpm))
    std_bpm = float(np.std(calibration_bpm))
    
    # Kategorisasi BPM (berdasarkan AHA)
    if avg_bpm < 60:
        hr_category = "Athletic"
        hr_interpretation = "Detak jantung sangat efisien, khas atlet"
        hr_score = 95
    elif avg_bpm < 70:
        hr_category = "Excellent"
        hr_interpretation = "Detak jantung sangat baik"
        hr_score = 90
    elif avg_bpm < 80:
        hr_category = "Normal"
        hr_interpretation = "Detak jantung dalam rentang normal"
        hr_score = 80
    elif avg_bpm < 90:
        hr_category = "Elevated"
        hr_interpretation = "Detak jantung sedikit tinggi, perlu perhatian"
        hr_score = 65
    else:
        hr_category = "High"
        hr_interpretation = "Detak jantung tinggi, disarankan konsultasi medis"
        hr_score = 50
    
    # === ANALISIS HRV ===
    hrv_values = []
    for i in range(len(calibration_bpm) - 4):
        window = calibration_bpm[i:i+5]
        hrv = calculate_hrv_features(window)
        if hrv:
            hrv_values.append({'sdnn': hrv['sdnn'], 'rmssd': hrv['rmssd']})
    
    if hrv_values:
        avg_sdnn = float(np.mean([h['sdnn'] for h in hrv_values]))
        avg_rmssd = float(np.mean([h['rmssd'] for h in hrv_values]))
    else:
        avg_sdnn = 0
        avg_rmssd = 0
    
    # Kategorisasi HRV (berdasarkan PMC 2017)
    if avg_sdnn > 5:
        hrv_category = "Excellent"
        hrv_interpretation = "Variabilitas jantung sangat baik, sistem saraf seimbang"
        hrv_score = 90
    elif avg_sdnn > 3:
        hrv_category = "Good"
        hrv_interpretation = "Variabilitas jantung baik"
        hrv_score = 75
    elif avg_sdnn > 1.5:
        hrv_category = "Fair"
        hrv_interpretation = "Variabilitas jantung cukup, tingkatkan dengan relaksasi"
        hrv_score = 60
    else:
        hrv_category = "Low"
        hrv_interpretation = "Variabilitas jantung rendah, pertimbangkan konsultasi"
        hrv_score = 45
    
    # === ANALISIS AKTIVITAS ===
    avg_motion = float(np.mean(calibration_motion))
    if avg_motion < 15:
        activity_during_mcu = "Resting"
    elif avg_motion < 40:
        activity_during_mcu = "Light Activity"
    else:
        activity_during_mcu = "Active"
    
    # === OVERALL SCORE ===
    overall_score = int(hr_score * 0.5 + hrv_score * 0.4 + 80 * 0.1)  # 10% bonus untuk partisipasi
    
    if overall_score >= 85:
        overall_status = "EXCELLENT"
        overall_message = "Kondisi kardiovaskular Anda sangat baik!"
    elif overall_score >= 70:
        overall_status = "GOOD"
        overall_message = "Kondisi kardiovaskular Anda baik, pertahankan!"
    elif overall_score >= 55:
        overall_status = "FAIR"
        overall_message = "Kondisi cukup baik, ada ruang untuk perbaikan"
    else:
        overall_status = "NEEDS ATTENTION"
        overall_message = "Disarankan konsultasi dengan profesional kesehatan"
    
    # === GENERATE REKOMENDASI PERSONAL ===
    recommendations = []
    
    if hr_category in ["Elevated", "High"]:
        recommendations.append("Lakukan kardio ringan 30 menit/hari (jalan cepat, bersepeda)")
        recommendations.append("Kurangi konsumsi kafein dan garam")
    elif hr_category == "Athletic":
        recommendations.append("Pertahankan rutinitas olahraga Anda, sangat baik!")
    
    if hrv_category in ["Fair", "Low"]:
        recommendations.append("Praktikkan teknik pernapasan 4-7-8 sebelum tidur")
        recommendations.append("Tingkatkan kualitas tidur 7-8 jam/malam")
    elif hrv_category == "Excellent":
        recommendations.append("Sistem saraf otonom Anda seimbang, pertahankan!")
    
    if std_bpm < 2:
        recommendations.append("Variabilitas detak sangat stabil, indikasi kesehatan baik")
    elif std_bpm > 8:
        recommendations.append("Perhatikan pola detak, pastikan istirahat cukup")
    
    if not recommendations:
        recommendations = [
            "Terus pertahankan pola hidup sehat",
            "Minum air putih minimal 8 gelas/hari",
            "Olahraga teratur 3-5 kali/minggu"
        ]
    
    # === INTERPRETASI LENGKAP ===
    interpretation = (
        f"Berdasarkan {len(calibration_bpm)} pembacaan selama proses kalibrasi, "
        f"detak jantung rata-rata Anda adalah {avg_bpm:.0f} BPM ({hr_category}), "
        f"dengan variabilitas (HRV) sebesar {avg_sdnn:.1f} ms ({hrv_category}). "
        f"{hr_interpretation}. {hrv_interpretation}."
    )
    
    # === BUILD REPORT ===
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": 90,
        "total_readings": len(calibration_bpm),
        "heart_rate": {
            "average": round(avg_bpm, 1),
            "min": round(min_bpm, 1),
            "max": round(max_bpm, 1),
            "std": round(std_bpm, 2),
            "category": hr_category,
            "interpretation": hr_interpretation,
            "score": hr_score
        },
        "hrv": {
            "sdnn_avg": round(avg_sdnn, 2),
            "rmssd_avg": round(avg_rmssd, 2),
            "category": hrv_category,
            "interpretation": hrv_interpretation,
            "score": hrv_score
        },
        "activity_during_mcu": activity_during_mcu,
        "overall_score": overall_score,
        "overall_status": overall_status,
        "overall_message": overall_message,
        "interpretation": interpretation,
        "recommendations": recommendations[:3],
        "disclaimer": "Hasil ini bersifat edukatif dan bukan pengganti diagnosis medis profesional. Konsultasikan dengan dokter untuk evaluasi kesehatan yang komprehensif."
    }
    
    print(f"\n[MCU] 📋 LAPORAN MCU LENGKAP untuk {device_id}")
    print(f"[MCU] Heart Rate: {avg_bpm:.0f} BPM ({hr_category})")
    print(f"[MCU] HRV: SDNN={avg_sdnn:.1f} ({hrv_category})")
    print(f"[MCU] Overall: {overall_score}/100 ({overall_status})")
    
    return report

def log_health_data(device_id, data):
    timestamp = datetime.now().isoformat()
    log_entry = {'timestamp': timestamp, 'device_id': device_id, **data}
    log_file = os.path.join(LOG_DIR, f"{device_id}_{datetime.now().strftime('%Y%m%d')}.jsonl")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, cls=NumpyEncoder) + '\n')

# ==========================================
# 🌐 DASHBOARD REAL-TIME BROADCAST
# ==========================================
dashboard_clients = []
dashboard_lock = threading.Lock()

def broadcast_to_dashboard(data):
    with dashboard_lock:
        dead_clients = []
        for client_queue in dashboard_clients:
            try:
                client_queue.put_nowait(data)
            except queue.Full:
                dead_clients.append(client_queue)
        for client in dead_clients:
            dashboard_clients.remove(client)

@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/stream')
def stream():
    def generate():
        client_queue = queue.Queue()
        with dashboard_lock:
            dashboard_clients.append(client_queue)
        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(data, cls=NumpyEncoder)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with dashboard_lock:
                if client_queue in dashboard_clients:
                    dashboard_clients.remove(client_queue)
    return app.response_class(generate(), mimetype='text/event-stream')

# ==========================================
# ENDPOINTS UTAMA
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        timestamp = datetime.now().strftime("%H:%M:%S")
        device_id = str(data.get('device_id', 'default'))
        bpm = float(data.get('bpm', 0))
        spo2 = float(data.get('spo2', 0))
        motion = float(data.get('motion_level', 0))
        
        print(f"\n[{timestamp}] Device: {device_id} | BPM: {bpm} | Motion: {motion}")
        
        if device_id not in device_data:
            device_data[device_id] = {'bpm_history': deque(maxlen=200), 'motion_history': deque(maxlen=200), 'total_readings': 0, 'models_trained': False}
        
        device_data[device_id]['bpm_history'].append(bpm)
        device_data[device_id]['motion_history'].append(motion)
        device_data[device_id]['total_readings'] += 1
        
        bpm_history = list(device_data[device_id]['bpm_history'])
        motion_history = list(device_data[device_id]['motion_history'])
        
        # FASE 1: TRAINING ML MODELS
        if not device_data[device_id]['models_trained'] and len(bpm_history) >= 30:
            print(f"\n[ML TRAINING] 🚀 Training 3 ML models untuk {device_id}...")
            rf_model, rf_info = train_activity_classifier(motion_history, bpm_history)
            km_model, km_scaler, km_info = train_health_clustering(bpm_history, motion_history)
            iso_model, iso_info = train_novelty_detector(bpm_history, motion_history)
            
            if rf_model and km_model and iso_model:
                device_models[device_id] = {'activity_rf': rf_model, 'health_kmeans': km_model, 'health_scaler': km_scaler, 'novelty_iso': iso_model, 'km_info': km_info}
                device_data[device_id]['models_trained'] = True
                print(f"[ML] ✅ RF: {rf_info} | K-Means: Trained | IsoForest: {iso_info}")
                joblib.dump(device_models[device_id], os.path.join(MODEL_DIR, f"{device_id}_models.pkl"))
                
                # ⭐ GENERATE MCU REPORT SAAT MODEL SELESAI TRAINING
                mcu_report = generate_mcu_report(device_id, bpm_history, motion_history)
                device_data[device_id]['mcu_report'] = mcu_report
                device_data[device_id]['mcu_ready'] = True
                print(f"[MCU] 📋 Laporan MCU siap ditampilkan ke user!")
        
        # FASE 2: ANALISIS
        recent_bpm = bpm_history[-WINDOW_SIZE:]
        hrv = calculate_hrv_features(recent_bpm)
        if not hrv:
            return jsonify({"status": "MONITORING", "action": f"Mengumpulkan data ({len(bpm_history)}/30)", "readings": device_data[device_id]['total_readings']}), 200
        
        # ML INFERENCE #1: Activity Recognition (HYBRID FALLBACK)
        activity_context = "REST"
        ml_activity_confidence = 0.0
        
        # Rule-based fallback (selalu tersedia)
        if motion < 15:
            rule_based_context = "REST"
        elif motion < 40:
            rule_based_context = "WALK"
        else:
            rule_based_context = "RUN"
        
        if device_id in device_models and 'activity_rf' in device_models[device_id]:
            rf = device_models[device_id]['activity_rf']
            features = np.array([[
                float(np.mean(motion_history[-WINDOW_SIZE:])),
                float(np.std(motion_history[-WINDOW_SIZE:])),
                hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd']
            ]])
            activity_pred = int(rf.predict(features)[0])
            activity_proba = rf.predict_proba(features)[0]
            ml_activity_confidence = float(np.max(activity_proba))
            
            activity_map = {0: "REST", 1: "WALK", 2: "RUN"}
            ml_predicted_context = activity_map[activity_pred]
            
            # ⭐ HYBRID LOGIC: Validasi prediksi ML dengan data real-time
            # Jika ML bilang REST tapi motion tinggi → override dengan rule-based
            if ml_predicted_context == "REST" and motion > 20:
                activity_context = rule_based_context
                print(f"[ML] ⚠️ Override: RF bilang REST, tapi motion={motion:.0f}. Pakai {rule_based_context}")
            elif ml_predicted_context == "REST" and rule_based_context != "REST" and motion > 15:
                activity_context = rule_based_context
                print(f"[ML] ⚠️ Override: RF bilang REST, tapi rule bilang {rule_based_context}. Motion={motion:.0f}")
            else:
                activity_context = ml_predicted_context
                
            print(f"[ML] 🏃 Activity (RF): {activity_context} (conf: {ml_activity_confidence:.2%}, motion: {motion:.0f})")
        else:
            # Model belum training, pakai rule-based
            activity_context = rule_based_context
            print(f"[RULE] 🏃 Activity (Rule-based): {activity_context} (motion: {motion:.0f})")
        
        # ML Inference 2: Clustering
        ml_health_cluster, ml_cluster_label = None, None
        if device_id in device_models and 'health_kmeans' in device_models[device_id]:
            km = device_models[device_id]['health_kmeans']
            scaler = device_models[device_id]['health_scaler']
            km_info = device_models[device_id]['km_info']
            features = np.array([[hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd'], float(np.mean(motion_history[-WINDOW_SIZE:]))]])
            cluster = int(km.predict(scaler.transform(features))[0])
            ml_health_cluster = cluster
            ml_cluster_label = str(km_info['cluster_labels'][cluster])
        
        # ML Inference 3: Novelty
        ml_novelty_score, ml_is_novel = 0.0, False
        if device_id in device_models and 'novelty_iso' in device_models[device_id]:
            iso = device_models[device_id]['novelty_iso']
            features = np.array([[hrv['mean_bpm'], hrv['sdnn'], hrv['rmssd'], float(np.mean(motion_history[-WINDOW_SIZE:]))]])
            novelty_pred = int(iso.predict(features)[0])
            ml_novelty_score = float(iso.score_samples(features)[0])
            ml_is_novel = bool(ml_novelty_score < -0.8 and novelty_pred == -1)
        
        # Rule-Based Health Score
        health_score, health_status = calculate_health_score(hrv['mean_bpm'], hrv['sdnn'], activity_context)
        
        message = "Kondisi kesehatan baik" if ml_cluster_label == "GOOD" else ("Pola kesehatan sangat baik!" if ml_cluster_label == "EXCELLENT" else "Perlu peningkatan kondisi fisik")
        recommendations = ["Terus pertahankan pola hidup sehat"] if not ml_is_novel else ["Pola detak tidak biasa, pastikan sensor menempel baik"]
        
        log_data = {
            'bpm': float(bpm), 'spo2': float(spo2), 'motion': float(motion), 'context': str(activity_context),
            'mean_bpm': float(hrv['mean_bpm']), 'sdnn': float(hrv['sdnn']), 'rmssd': float(hrv['rmssd']),
            'health_score': int(health_score), 'health_status': str(health_status), 'ml_is_novel': bool(ml_is_novel)
        }
        log_health_data(device_id, log_data)
        
        print(f"[RESULT] Score: {health_score} ({health_status}) | Context: {activity_context}")
        
        # BROADCAST DATA KE DASHBOARD SEBELUM RETURN
        broadcast_data = {
            "metrics": {"mean_bpm": round(float(hrv['mean_bpm']), 1), "sdnn": round(float(hrv['sdnn']), 2), "rmssd": round(float(hrv['rmssd']), 2)},
            "health_score": int(health_score), "health_status": str(health_status), "message": str(message), "context": str(activity_context),
            "ml_analysis": {
                "activity_recognition": {"confidence": round(float(ml_activity_confidence), 3)} if ml_activity_confidence > 0 else None,
                "novelty_detection": {"score": round(float(ml_novelty_score), 3)}
            },
            "recommendations": recommendations,
            "models_trained": bool(device_data[device_id]['models_trained']),
            "readings": int(device_data[device_id]['total_readings'])
        }
        
        # ⭐ KIRIM MCU REPORT JIKA BARU SELESAI TRAINING
        if device_data[device_id].get('mcu_ready'):
            broadcast_data["mcu_report"] = device_data[device_id]['mcu_report']
            broadcast_data["show_mcu"] = True
            device_data[device_id]['mcu_ready'] = False  # Reset flag, hanya tampil sekali
        
        # Info retrain (jika ada)
        if device_data[device_id].get('retrain_triggered') and not device_data[device_id]['models_trained']:
            broadcast_data["retrain_status"] = "in_progress"
            broadcast_data["progress"] = min(100, int((device_data[device_id]['total_readings'] / 30) * 100))
            broadcast_data["target"] = 30
        elif device_data[device_id].get('retrain_triggered') and device_data[device_id]['models_trained']:
            broadcast_data["retrain_status"] = "completed"
            device_data[device_id]['retrain_triggered'] = False
        
        broadcast_to_dashboard(broadcast_data)
        
        return jsonify({
            "status": "MONITORING", "health_score": int(health_score), "health_status": str(health_status),
            "message": str(message), "context": str(activity_context),
            "metrics": {"mean_bpm": round(float(hrv['mean_bpm']), 1), "sdnn": round(float(hrv['sdnn']), 2), "rmssd": round(float(hrv['rmssd']), 2)},
            "ml_analysis": {
                "activity_recognition": {"method": "Random Forest", "prediction": str(activity_context), "confidence": round(float(ml_activity_confidence), 3)},
                "health_clustering": {"method": "K-Means", "cluster": int(ml_health_cluster) if ml_health_cluster is not None else None, "label": str(ml_cluster_label) if ml_cluster_label else None},
                "novelty_detection": {"method": "Isolation Forest", "score": round(float(ml_novelty_score), 3), "is_novel": bool(ml_is_novel)}
            },
            "recommendations": recommendations, "models_trained": bool(device_data[device_id]['models_trained']), "readings": int(device_data[device_id]['total_readings'])
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "ERROR", "action": "Server error", "error": str(e)}), 400
    
    
    
@app.route('/retrain/<device_id>', methods=['POST'])
def retrain_model(device_id):
    """
    Endpoint untuk melatih ulang model ML dengan user baru.
    Akan mereset semua data historis dan model yang sudah dilatih.
    """
    try:
        device_id = str(device_id)
        
        # Reset device data
        if device_id in device_data:
            device_data[device_id] = {
                'bpm_history': deque(maxlen=200),
                'motion_history': deque(maxlen=200),
                'total_readings': 0,
                'models_trained': False,
                'retrain_triggered': True,
                'retrain_start_time': datetime.now().isoformat()
            }
        else:
            device_data[device_id] = {
                'bpm_history': deque(maxlen=200),
                'motion_history': deque(maxlen=200),
                'total_readings': 0,
                'models_trained': False,
                'retrain_triggered': True,
                'retrain_start_time': datetime.now().isoformat()
            }
        
        # Hapus model lama dari memory
        if device_id in device_models:
            del device_models[device_id]
        
        # Hapus file model lama dari disk
        model_file = os.path.join(MODEL_DIR, f"{device_id}_models.pkl")
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"[RETRAIN] 🗑️ Model lama dihapus: {model_file}")
        
        # Broadcast status retraining ke dashboard
        broadcast_to_dashboard({
            "retrain_status": "started",
            "device_id": device_id,
            "message": "Model sedang dilatih ulang untuk user baru...",
            "progress": 0,
            "readings": 0,
            "target": 30
        })
        
        print(f"\n[RETRAIN] 🔄 Model untuk {device_id} telah direset!")
        print(f"[RETRAIN] ⏳ Menunggu 30 data point baru untuk training ulang...")
        
        return jsonify({
            "status": "success",
            "message": f"Model untuk {device_id} telah direset. Silakan tempelkan jari untuk kalibrasi ulang (~90 detik).",
            "device_id": device_id
        }), 200
        
    except Exception as e:
        print(f"[RETRAIN ERROR] {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/status/<device_id>', methods=['GET'])
def get_device_status(device_id):
    """Endpoint untuk cek status device (digunakan dashboard untuk polling)"""
    device_id = str(device_id)
    
    if device_id not in device_data:
        return jsonify({
            "exists": False,
            "models_trained": False,
            "readings": 0,
            "target": 30
        }), 200
    
    data = device_data[device_id]
    readings = data['total_readings']
    
    return jsonify({
        "exists": True,
        "models_trained": data['models_trained'],
        "readings": readings,
        "target": 30,
        "progress": min(100, int((readings / 30) * 100)),
        "retrain_triggered": data.get('retrain_triggered', False),
        "retrain_start_time": data.get('retrain_start_time', None)
    }), 200

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 SERVER AI: Smart Heart Rate Health Monitor")
    print("📡 Dashboard: http://localhost:5000/dashboard")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)