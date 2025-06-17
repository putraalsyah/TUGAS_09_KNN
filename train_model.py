import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime

class IrisKNNTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        self.training_history = {}
        
    def load_and_prepare_data(self):
        """Load dan prepare dataset Iris"""
        print("Loading and preparing Iris dataset...")
        
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Create DataFrame untuk analisis
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = y
        df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
        
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Features: {feature_names}")
        print(f"Classes: {df['species_name'].unique()}")
        print(f"Class distribution:\n{df['species_name'].value_counts()}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisasi data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nData split:")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
        return df
    
    def hyperparameter_tuning(self):
        """Tuning hyperparameter menggunakan GridSearch"""
        print("\nPerforming hyperparameter tuning...")
        
        # Parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # GridSearch
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, use_tuning=True):
        """Train model KNN"""
        print("\nTraining KNN model...")
        
        if use_tuning:
            self.model = self.hyperparameter_tuning()
        else:
            # Gunakan parameter default yang baik
            self.model = KNeighborsClassifier(
                n_neighbors=5, 
                weights='uniform', 
                metric='euclidean'
            )
            self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        return self.model
    
    def evaluate_model(self):
        """Evaluasi model"""
        print("\nEvaluating model...")
        
        # Prediksi
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred, target_names=target_names))
        
        # Store training history
        self.training_history = {
            'timestamp': datetime.now().isoformat(),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'model_params': self.best_params if self.best_params else self.model.get_params(),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        return test_accuracy
    
    def test_sample_predictions(self):
        """Test prediksi dengan sample data"""
        print("\nTesting sample predictions...")
        
        # Sample data untuk testing
        test_samples = [
            {'features': [5.1, 3.5, 1.4, 0.2], 'expected': 'Setosa'},
            {'features': [7.0, 3.2, 4.7, 1.4], 'expected': 'Versicolor'},
            {'features': [6.3, 3.3, 6.0, 2.5], 'expected': 'Virginica'},
            {'features': [4.9, 3.0, 1.4, 0.2], 'expected': 'Setosa'},
            {'features': [6.7, 3.1, 4.4, 1.4], 'expected': 'Versicolor'},
        ]
        
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        
        print(f"{'No':<3} {'Features':<25} {'Predicted':<12} {'Expected':<12} {'Confidence':<10} {'Match'}")
        print("-" * 80)
        
        correct_predictions = 0
        
        for i, sample in enumerate(test_samples):
            # Scale features
            features_scaled = self.scaler.transform([sample['features']])
            
            # Prediksi
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities) * 100
            
            predicted_class = target_names[prediction]
            expected_class = sample['expected']
            match = "✓" if predicted_class == expected_class else "✗"
            
            if predicted_class == expected_class:
                correct_predictions += 1
            
            print(f"{i+1:<3} {str(sample['features']):<25} {predicted_class:<12} {expected_class:<12} {confidence:<10.1f}% {match}")
        
        sample_accuracy = correct_predictions / len(test_samples) * 100
        print(f"\nSample Test Accuracy: {sample_accuracy:.1f}% ({correct_predictions}/{len(test_samples)})")
    
    def save_model(self):
        """Simpan model dan metadata"""
        print("\nSaving model and metadata...")
        
        try:
            # Simpan model dan scaler
            joblib.dump(self.model, 'knn_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            
            # Simpan training history
            with open('training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Simpan model info
            model_info = {
                'algorithm': 'K-Nearest Neighbors',
                'sklearn_version': '1.0+',
                'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                'classes': ['Setosa', 'Versicolor', 'Virginica'],
                'model_params': self.model.get_params(),
                'scaler_type': 'StandardScaler',
                'dataset': 'Iris Flower Dataset'
            }
            
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print("Files saved successfully:")
            print("- knn_model.pkl (trained model)")
            print("- scaler.pkl (feature scaler)")
            print("- training_history.json (training metrics)")
            print("- model_info.json (model information)")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load model yang sudah tersimpan"""
        try:
            self.model = joblib.load('knn_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            
            with open('model_info.json', 'r') as f:
                model_info = json.load(f)
            
            print("Model loaded successfully!")
            print(f"Algorithm: {model_info['algorithm']}")
            print(f"Parameters: {model_info['model_params']}")
            
            return True
            
        except FileNotFoundError:
            print("No saved model found. Need to train new model.")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def run_training_pipeline(self, use_tuning=True):
        """Jalankan pipeline training lengkap"""
        print("="*60)
        print("IRIS FLOWER CLASSIFICATION - KNN MODEL TRAINING")
        print("="*60)
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Train model
        self.train_model(use_tuning=use_tuning)
        
        # Evaluate
        accuracy = self.evaluate_model()
        
        # Test samples
        self.test_sample_predictions()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        if self.best_params:
            print(f"Best Parameters: {self.best_params}")
        print("="*60)
        
        return accuracy

def main():
    """Main function"""
    trainer = IrisKNNTrainer()
    
    # Cek apakah sudah ada model
    if trainer.load_model():
        print("\nExisting model found!")
        response = input("Do you want to retrain the model? (y/n): ")
        if response.lower() != 'y':
            print("Using existing model.")
            trainer.test_sample_predictions()
            return
    
    # Training model baru
    print("\nStarting training process...")
    use_tuning = input("Use hyperparameter tuning? (y/n, default=y): ").lower() != 'n'
    
    trainer.run_training_pipeline(use_tuning=use_tuning)

if __name__ == "__main__":
    main()