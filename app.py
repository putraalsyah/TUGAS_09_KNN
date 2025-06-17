from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

app = Flask(__name__)

# Global variables untuk model
model = None
scaler = None
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = ['Setosa', 'Versicolor', 'Virginica']

def load_or_train_model():
    """Load model yang sudah ada atau train model baru"""
    global model, scaler
    
    # Cek apakah model sudah ada
    if os.path.exists('knn_model.pkl') and os.path.exists('scaler.pkl'):
        print("Loading existing model...")
        model = joblib.load('knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return True
    else:
        print("Training new model...")
        return train_new_model()

def train_new_model():
    """Train model KNN baru dengan dataset Iris"""
    global model, scaler
    
    try:
        # Load dataset Iris (buat dataset sendiri jika tidak ada sklearn.datasets)
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisasi data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model KNN dengan k=5
        model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        model.fit(X_train_scaled, y_train)
        
        # Evaluasi model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Simpan model
        joblib.dump(model, 'knn_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    """Prediksi jenis bunga berdasarkan input features"""
    global model, scaler
    
    try:
        # Buat array input
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale input
        input_scaled = scaler.transform(input_features)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Get distances to nearest neighbors
        distances, indices = model.kneighbors(input_scaled, n_neighbors=5)
        
        result = {
            'prediction': target_names[prediction],
            'confidence': round(confidence, 2),
            'probabilities': {
                target_names[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probability)
            },
            'distances': distances[0].tolist()
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint untuk prediksi"""
    try:
        # Ambil data dari form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Validasi input
        if not all(0 < x < 20 for x in [sepal_length, sepal_width, petal_length, petal_width]):
            return jsonify({'error': 'Input values must be between 0 and 20'})
        
        # Prediksi
        result = predict_flower(sepal_length, sepal_width, petal_length, petal_width)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except ValueError:
        return jsonify({'error': 'Please enter valid numeric values'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """Informasi tentang model"""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    info = {
        'algorithm': 'K-Nearest Neighbors (KNN)',
        'n_neighbors': model.n_neighbors,
        'metric': model.metric,
        'features': feature_names,
        'classes': target_names,
        'n_features': len(feature_names),
        'n_classes': len(target_names)
    }
    
    return jsonify(info)

@app.route('/about')
def about():
    """Halaman tentang aplikasi"""
    return render_template('about.html')

if __name__ == '__main__':
    print("Starting Iris Flower Classification App...")
    print("Loading/Training KNN Model...")
    
    # Load atau train model
    if load_or_train_model():
        print("Model ready!")
        print(f"Model info: K={model.n_neighbors}, Metric={model.metric}")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load/train model. Exiting...")