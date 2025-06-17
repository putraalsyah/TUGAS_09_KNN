import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
import json

class IrisDataGenerator:
    def __init__(self):
        self.iris_data = None
        self.df = None
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = ['Setosa', 'Versicolor', 'Virginica']
    
    def load_iris_dataset(self):
        """Load dataset Iris dari sklearn"""
        print("Loading Iris dataset from sklearn...")
        
        # Load dataset
        self.iris_data = load_iris()
        X, y = self.iris_data.data, self.iris_data.target
        
        # Buat DataFrame
        self.df = pd.DataFrame(X, columns=self.feature_names)
        self.df['species'] = y
        self.df['species_name'] = self.df['species'].map({
            0: 'Setosa', 
            1: 'Versicolor', 
            2: 'Virginica'
        })
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Classes: {self.target_names}")
        
        return self.df
    
    def analyze_dataset(self):
        """Analisis dataset Iris"""
        print("\n" + "="*50)
        print("IRIS DATASET ANALYSIS")
        print("="*50)
        
        if self.df is None:
            self.load_iris_dataset()
        
        # Basic info
        print("\n1. BASIC INFORMATION:")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of classes: {len(self.target_names)}")
        
        # Class distribution
        print("\n2. CLASS DISTRIBUTION:")
        class_counts = self.df['species_name'].value_counts()
        for species, count in class_counts.items():
            print(f"{species}: {count} samples ({count/len(self.df)*100:.1f}%)")
        
        # Statistical summary
        print("\n3. STATISTICAL SUMMARY:")
        print(self.df[self.feature_names].describe().round(2))
        
        # Missing values
        print("\n4. MISSING VALUES:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        return self.df.describe()
    
    def visualize_dataset(self):
        """Visualisasi dataset"""
        print("\nGenerating dataset visualizations...")
        
        if self.df is None:
            self.load_iris_dataset()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Iris Dataset - Feature Distributions by Species', fontsize=16)
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i//2, i%2]
            
            for species in self.target_names:
                species_data = self.df[self.df['species_name'] == species][feature]
                ax.hist(species_data, alpha=0.7, label=species, bins=15)
            
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('iris_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[self.feature_names].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Pairplot
        plt.figure(figsize=(12, 10))
        sns.pairplot(self.df, hue='species_name', vars=self.feature_names, 
                    diag_kind='hist', markers=['o', 's', 'D'])
        plt.suptitle('Iris Dataset - Pairwise Feature Relationships', y=1.02)
        plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Box plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Iris Dataset - Feature Box Plots by Species', fontsize=16)
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i//2, i%2]
            sns.boxplot(data=self.df, x='species_name', y=feature, ax=ax)
            ax.set_title(f'{feature.replace("_", " ").title()} by Species')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved:")
        print("- iris_distributions.png")
        print("- iris_correlation.png") 
        print("- iris_pairplot.png")
        print("- iris_boxplots.png")
    
    def generate_sample_data(self, n_samples=10):
        """Generate sample data untuk testing"""
        print(f"\nGenerating {n_samples} sample data points...")
        
        if self.df is None:
            self.load_iris_dataset()
        
        # Ambil statistik dari dataset asli
        stats = {}
        for feature in self.feature_names:
            stats[feature] = {
                'mean': self.df[feature].mean(),
                'std': self.df[feature].std(),
                'min': self.df[feature].min(),
                'max': self.df[feature].max()
            }
        
        # Generate sample data berdasarkan distribusi setiap class
        samples = []
        
        for species_idx, species in enumerate(self.target_names):
            species_data = self.df[self.df['species_name'] == species]
            
            for i in range(n_samples // 3):
                sample = {}
                sample['id'] = len(samples) + 1
                
                # Generate features berdasarkan mean dan std dari species
                for feature in self.feature_names:
                    mean = species_data[feature].mean()
                    std = species_data[feature].std()
                    
                    # Add some randomness
                    value = np.random.normal(mean, std * 0.5)
                    
                    # Ensure realistic bounds
                    min_val = stats[feature]['min']
                    max_val = stats[feature]['max']
                    value = np.clip(value, min_val, max_val)
                    
                    sample[feature] = round(value, 1)
                
                sample['actual_species'] = species
                samples.append(sample)
        
        # Create DataFrame
        sample_df = pd.DataFrame(samples)
        
        # Save to CSV
        sample_df.to_csv('sample_test_data.csv', index=False)
        
        print("Sample data generated and saved to 'sample_test_data.csv'")
        print(f"Generated {len(samples)} samples")
        print("\nSample preview:")
        print(sample_df.head(10))
        
        return sample_df
    
    def create_dataset_info(self):
        """Buat file informasi dataset"""
        print("\nCreating dataset information file...")
        
        if self.df is None:
            self.load_iris_dataset()
        
        # Statistik per class
        class_stats = {}
        for species in self.target_names:
            species_data = self.df[self.df['species_name'] == species]
            class_stats[species] = {
                'count': len(species_data),
                'features': {}
            }
            
            for feature in self.feature_names:
                class_stats[species]['features'][feature] = {
                    'mean': float(species_data[feature].mean()),
                    'std': float(species_data[feature].std()),
                    'min': float(species_data[feature].min()),
                    'max': float(species_data[feature].max())
                }
        
        # Dataset info
        dataset_info = {
            'name': 'Iris Flower Dataset',
            'description': 'The Iris flower data set is a multivariate data set introduced by Ronald Fisher in 1936',
            'total_samples': len(self.df),
            'n_features': len(self.feature_names),
            'n_classes': len(self.target_names),
            'features': {
                'names': self.feature_names,
                'descriptions': {
                    'sepal_length': 'Length of the sepal in cm',
                    'sepal_width': 'Width of the sepal in cm', 
                    'petal_length': 'Length of the petal in cm',
                    'petal_width': 'Width of the petal in cm'
                }
            },
            'classes': {
                'names': self.target_names,
                'descriptions': {
                    'Setosa': 'Iris Setosa species',
                    'Versicolor': 'Iris Versicolor species',
                    'Virginica': 'Iris Virginica species'
                }
            },
            'class_statistics': class_stats,
            'use_cases': [
                'Classification problem',
                'Pattern recognition',
                'Machine learning tutorials',
                'Statistical analysis'
            ]
        }
        
        # Save to JSON
        with open('dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print("Dataset information saved to 'dataset_info.json'")
        
        return dataset_info
    
    def export_dataset(self, format='csv'):
        """Export dataset ke berbagai format"""
        print(f"\nExporting dataset to {format.upper()} format...")
        
        if self.df is None:
            self.load_iris_dataset()
        
        if format.lower() == 'csv':
            self.df.to_csv('iris_dataset.csv', index=False)
            print("Dataset exported to 'iris_dataset.csv'")
        
        elif format.lower() == 'json':
            self.df.to_json('iris_dataset.json', orient='records', indent=2)
            print("Dataset exported to 'iris_dataset.json'")
        
        elif format.lower() == 'excel':
            with pd.ExcelWriter('iris_dataset.xlsx') as writer:
                self.df.to_excel(writer, sheet_name='iris_data', index=False)
                
                # Add summary sheet
                summary = self.df.groupby('species_name')[self.feature_names].describe()
                summary.to_excel(writer, sheet_name='summary')
            
            print("Dataset exported to 'iris_dataset.xlsx'")
        
        else:
            print(f"Format '{format}' not supported. Use 'csv', 'json', or 'excel'")
    
    def run_complete_analysis(self):
        """Jalankan analisis lengkap"""
        print("="*60)
        print("IRIS DATASET - COMPLETE ANALYSIS")
        print("="*60)
        
        # Load dataset
        self.load_iris_dataset()
        
        # Analyze
        self.analyze_dataset()
        
        # Visualize
        self.visualize_dataset()
        
        # Generate samples
        self.generate_sample_data()
        
        # Create info
        self.create_dataset_info()
        
        # Export
        self.export_dataset('csv')
        self.export_dataset('json')
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED!")
        print("Generated files:")
        print("- iris_dataset.csv")
        print("- iris_dataset.json") 
        print("- dataset_info.json")
        print("- sample_test_data.csv")
        print("- Various visualization PNG files")
        print("="*60)

def main():
    """Main function"""
    generator = IrisDataGenerator()
    
    print("Iris Dataset Generator and Analyzer")
    print("1. Complete analysis (recommended)")
    print("2. Basic analysis only")
    print("3. Generate visualizations only")
    print("4. Generate sample data only")
    
    choice = input("Enter your choice (1-4, default=1): ").strip()
    
    if choice == '2':
        generator.load_iris_dataset()
        generator.analyze_dataset()
    elif choice == '3':
        generator.load_iris_dataset()
        generator.visualize_dataset()
    elif choice == '4':
        generator.load_iris_dataset()
        generator.generate_sample_data()
    else:
        generator.run_complete_analysis()

if __name__ == "__main__":
    main()