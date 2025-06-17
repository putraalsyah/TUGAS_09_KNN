import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelUtils:
    """Utility functions untuk model management"""
    
    @staticmethod
    def validate_input(sepal_length, sepal_width, petal_length, petal_width):
        """Validasi input features"""
        try:
            # Convert to float
            sl = float(sepal_length)
            sw = float(sepal_width)
            pl = float(petal_length)
            pw = float(petal_width)
            
            # Check realistic ranges for iris flowers
            if not (0.5 <= sl <= 10.0):
                return False, "Sepal length must be between 0.5 and 10.0 cm"
            if not (0.5 <= sw <= 6.0):
                return False, "Sepal width must be between 0.5 and 6.0 cm"
            if not (0.1 <= pl <= 8.0):
                return False, "Petal length must be between 0.1 and 8.0 cm"
            if not (0.1 <= pw <= 4.0):
                return False, "Petal width must be between 0.1 and 4.0 cm"
            
            return True, [sl, sw, pl, pw]
            
        except ValueError:
            return False, "All inputs must be valid numbers"
    
    @staticmethod
    def get_feature_ranges():
        """Get typical feature ranges for Iris dataset"""
        return {
            'sepal_length': {'min': 4.3, 'max': 7.9, 'mean': 5.84},
            'sepal_width': {'min': 2.0, 'max': 4.4, 'mean': 3.06},
            'petal_length': {'min': 1.0, 'max': 6.9, 'mean': 3.76},
            'petal_width': {'min': 0.1, 'max': 2.5, 'mean': 1.20}
        }
    
    @staticmethod
    def calculate_confidence_level(probabilities):
        """Calculate confidence level from probabilities"""
        max_prob = max(probabilities)
        
        if max_prob >= 0.9:
            return "Very High"
        elif max_prob >= 0.7:
            return "High"
        elif max_prob >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def format_prediction_result(prediction, probabilities, distances=None):
        """Format prediction result for display"""
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        
        result = {
            'predicted_class': target_names[prediction],
            'confidence': round(max(probabilities) * 100, 2),
            'confidence_level': ModelUtils.calculate_confidence_level(probabilities),
            'class_probabilities': {
                target_names[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        if distances is not None:
            result['nearest_neighbors_distances'] = [round(d, 4) for d in distances]
        
        return result

class DataUtils:
    """Utility functions untuk data processing"""
    
    @staticmethod
    def load_sample_data(file_path='sample_test_data.csv'):
        """Load sample test data"""
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            return None
    
    @staticmethod
    def generate_random_sample():
        """Generate random sample for testing"""
        ranges = ModelUtils.get_feature_ranges()
        
        sample = {}
        for feature, range_info in ranges.items():
            # Generate value within typical range with some variance
            mean = range_info['mean']
            std = (range_info['max'] - range_info['min']) * 0.15
            value = np.random.normal(mean, std)
            
            # Clip to realistic bounds
            value = np.clip(value, range_info['min'], range_info['max'])
            sample[feature] = round(value, 1)
        
        return sample
    
    @staticmethod
    def create_feature_statistics():
        """Create feature statistics for reference"""
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        X = iris.data
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        stats = {}
        for i, feature in enumerate(feature_names):
            stats[feature] = {
                'mean': float(np.mean(X[:, i])),
                'std': float(np.std(X[:, i])),
                'min': float(np.min(X[:, i])),
                'max': float(np.max(X[:, i])),
                'q25': float(np.percentile(X[:, i], 25)),
                'q75': float(np.percentile(X[:, i], 75))
            }
        
        return stats

class LoggingUtils:
    """Utility functions untuk logging dan monitoring"""
    
    @staticmethod
    def log_prediction(features, prediction, confidence, timestamp=None):
        """Log prediction untuk monitoring"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'features': {
                'sepal_length': features[0],
                'sepal_width': features[1],
                'petal_length': features[2],
                'petal_width': features[3]
            },
            'prediction': prediction,
            'confidence': confidence
        }
        
        # Append to log file
        log_file = 'prediction_logs.json'
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    @staticmethod
    def get_prediction_stats():
        """Get prediction statistics"""
        log_file = 'prediction_logs.json'
        
        if not os.path.exists(log_file):
            return {'total_predictions': 0}
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {'total_predictions': 0}
            
            # Calculate statistics
            total = len(logs)
            predictions = [log['prediction'] for log in logs]
            confidences = [log['confidence'] for log in logs]
            
            prediction_counts = {}
            for pred in predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            stats = {
                'total_predictions': total,
                'prediction_distribution': prediction_counts,
                'average_confidence': round(np.mean(confidences), 2),
                'confidence_std': round(np.std(confidences), 2),
                'last_prediction_time': logs[-1]['timestamp'] if logs else None
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting prediction stats: {str(e)}")
            return {'total_predictions': 0, 'error': str(e)}

class ModelEvaluator:
    """Utility untuk evaluasi model"""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.target_names = ['Setosa', 'Versicolor', 'Virginica']
    
    def evaluate_on_test_data(self, X_test, y_test):
        """Evaluate model on test data"""
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.target_names):
            class_indices = (y_test == i)
            if np.any(class_indices):
                per_class_metrics[class_name] = {
                    'precision': precision_score(y_test == i, y_pred == i),
                    'recall': recall_score(y_test == i, y_pred == i),
                    'f1_score': f1_score(y_test == i, y_pred == i)
                }
        
        return {
            'overall_metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist()
        }
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation"""
        from sklearn.model_selection import cross_val_score
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_folds': cv
        }

class FileUtils:
    """Utility functions untuk file operations"""
    
    @staticmethod
    def check_model_files():
        """Check if model files exist"""
        required_files = ['knn_model.pkl', 'scaler.pkl']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        return len(missing_files) == 0, missing_files
    
    @staticmethod
    def get_model_info():
        """Get model information"""
        info_file = 'model_info.json'
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        
        return None
    
    @staticmethod
    def save_app_config(config):
        """Save application configuration"""
        with open('app_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_app_config():
        """Load application configuration"""
        config_file = 'app_config.json'
        
        default_config = {
            'model_path': 'knn_model.pkl',
            'scaler_path': 'scaler.pkl',
            'log_predictions': True,
            'max_log_entries': 1000,
            'debug_mode': False
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except:
                return default_config
        
        return default_config

# Helper functions
def format_number(num, decimals=2):
    """Format number for display"""
    return round(float(num), decimals)

def get_species_info():
    """Get information about iris species"""
    return {
        'Setosa': {
            'description': 'Iris setosa is a species of blue flag commonly known as the bristle-pointed iris.',
            'characteristics': ['Smaller petals', 'Distinct sepal patterns', 'Usually blue/purple flowers'],
            'typical_ranges': {
                'sepal_length': '4.3-5.8 cm',
                'sepal_width': '2.3-4.4 cm', 
                'petal_length': '1.0-1.9 cm',
                'petal_width': '0.1-0.6 cm'
            }
        },
        'Versicolor': {
            'description': 'Iris versicolor is also commonly known as the blue flag, harlequin blueflag, or wild iris.',
            'characteristics': ['Medium-sized features', 'Purple/blue flowers', 'Moderate petal size'],
            'typical_ranges': {
                'sepal_length': '4.9-7.0 cm',
                'sepal_width': '2.0-3.4 cm',
                'petal_length': '3.0-5.1 cm', 
                'petal_width': '1.0-1.8 cm'
            }
        },
        'Virginica': {
            'description': 'Iris virginica is commonly known as the Virginia iris, southern blue flag, or great blue flag.',
            'characteristics': ['Largest features', 'Blue/purple flowers', 'Longest petals'],
            'typical_ranges': {
                'sepal_length': '4.9-7.9 cm',
                'sepal_width': '2.2-3.8 cm',
                'petal_length': '4.5-6.9 cm',
                'petal_width': '1.4-2.5 cm'
            }
        }
    }

def create_sample_inputs():
    """Create sample inputs for testing"""
    return [
        {
            'name': 'Typical Setosa',
            'features': [5.1, 3.5, 1.4, 0.2],
            'description': 'Classic Setosa measurements'
        },
        {
            'name': 'Typical Versicolor', 
            'features': [7.0, 3.2, 4.7, 1.4],
            'description': 'Classic Versicolor measurements'
        },
        {
            'name': 'Typical Virginica',
            'features': [6.3, 3.3, 6.0, 2.5], 
            'description': 'Classic Virginica measurements'
        },
        {
            'name': 'Boundary Case 1',
            'features': [6.0, 3.0, 4.0, 1.3],
            'description': 'Between Versicolor and Virginica'
        },
        {
            'name': 'Small Flower',
            'features': [4.5, 2.5, 1.2, 0.3],
            'description': 'Small measurements, likely Setosa'
        }
    ]