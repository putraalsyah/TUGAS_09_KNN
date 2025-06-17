import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class IrisKNNTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = ['Setosa', 'Versicolor', 'Virginica']
    
    def load_data(self):
        """Load dan prepare dataset Iris"""
        print("Loading Iris dataset...")
        
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisasi
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {self.target_names}")
    
    def find_optimal_k(self, max_k=20):
        """Cari nilai k optimal menggunakan cross-validation"""
        print("\nFinding optimal k value...")
        
        k_values = range(1, max_k + 1)
        cv_scores = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
        
        # Plot hasil
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, cv_scores, marker='o', linewidth=2, markersize=6)
        plt.title('K-NN Cross Validation Scores')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross Validation Accuracy')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Highlight optimal k
        optimal_k = k_values[np.argmax(cv_scores)]
        optimal_score = max(cv_scores)
        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
        plt.text(optimal_k + 0.5, optimal_score - 0.01, 
                f'Optimal k={optimal_k}\nAccuracy={optimal_score:.4f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('optimal_k_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Optimal k: {optimal_k}")
        print(f"Best CV accuracy: {optimal_score:.4f}")
        
        return optimal_k
    
    def train_model(self, k=None):
        """Train model KNN"""
        if k is None:
            k = self.find_optimal_k()
        
        print(f"\nTraining KNN model with k={k}...")
        
        self.model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        self.model.fit(self.X_train, self.y_train)
        
        print("Model trained successfully!")
        return self.model
    
    def evaluate_model(self):
        """Evaluasi performa model"""
        print("\nEvaluating model performance...")
        
        # Prediksi
        y_pred = self.model.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def test_predictions(self):
        """Test beberapa prediksi sample"""
        print("\nTesting sample predictions...")
        
        # Test samples (sepal_length, sepal_width, petal_length, petal_width)
        test_samples = [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [7.0, 3.2, 4.7, 1.4],  # Versicolor
            [6.3, 3.3, 6.0, 2.5],  # Virginica
            [4.9, 3.0, 1.4, 0.2],  # Setosa
            [5.9, 3.0, 5.1, 1.8],  # Virginica
        ]
        
        expected = ['Setosa', 'Versicolor', 'Virginica', 'Setosa', 'Virginica']
        
        print(f"{'Features':<25} {'Predicted':<12} {'Expected':<12} {'Probabilities'}")
        print("-" * 80)
        
        for i, sample in enumerate(test_samples):
            # Scale sample
            sample_scaled = self.scaler.transform([sample])
            
            # Prediksi
            prediction = self.model.predict(sample_scaled)[0]
            probabilities = self.model.predict_proba(sample_scaled)[0]
            
            # Format output
            features_str = f"{sample}"
            predicted_class = self.target_names[prediction]
            expected_class = expected[i]
            prob_str = f"{probabilities}"
            
            print(f"{features_str:<25} {predicted_class:<12} {expected_class:<12} {prob_str}")
    
    def save_model(self):
        """Simpan model yang telah dilatih"""
        print("\nSaving trained model...")
        
        joblib.dump(self.model, 'knn_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        print("Model saved successfully!")
        print("- knn_model.pkl")
        print("- scaler.pkl")
    
    def run_complete_test(self):
        """Jalankan semua test"""
        print("="*60)
        print("IRIS FLOWER CLASSIFICATION WITH KNN")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model()
        
        # Evaluate
        accuracy = self.evaluate_model()
        
        # Test predictions
        self.test_predictions()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"Final Model Accuracy: {accuracy:.4f}")
        print("="*60)

def main():
    """Main function untuk menjalankan testing"""
    tester = IrisKNNTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main()