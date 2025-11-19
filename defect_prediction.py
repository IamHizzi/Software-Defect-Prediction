"""
Defect Prediction Module
Phase 1: ML-based defect prediction with ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import warnings
warnings.filterwarnings('ignore')


class DefectPredictor:
    """
    Ensemble-based defect prediction with feature selection and class balancing
    """
    
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.sampler = SMOTETomek(random_state=42)
        self.ensemble = None
        self.selected_features = None
        
    def build_ensemble(self):
        """Build ensemble voting classifier"""
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        dt = DecisionTreeClassifier(max_depth=8, random_state=42)
        
        self.ensemble = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm), ('dt', dt)],
            voting='soft',
            weights=[2, 1, 1]
        )
        
    def select_features(self, X, y):
        """Feature selection using mutual information"""
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif, 
            k=min(self.n_features, X.shape[1])
        )
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Store selected feature indices
        self.selected_features = self.feature_selector.get_support(indices=True)
        return X_selected
    
    def balance_classes(self, X, y):
        """Apply SMOTE-Tomek for class balancing"""
        try:
            X_balanced, y_balanced = self.sampler.fit_resample(X, y)
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Warning: Could not apply SMOTE-Tomek: {e}")
            return X, y
    
    def train(self, X, y):
        """Train the defect prediction model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        X_selected = self.select_features(X_scaled, y)
        
        # Balance classes
        X_balanced, y_balanced = self.balance_classes(X_selected, y)
        
        # Build and train ensemble
        self.build_ensemble()
        self.ensemble.fit(X_balanced, y_balanced)
        
        return self
    
    def predict(self, X):
        """Predict defect probability for modules"""
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        predictions = self.ensemble.predict(X_selected)
        probabilities = self.ensemble.predict_proba(X_selected)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary', zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return results


def extract_software_metrics(code_text):
    """
    Extract software metrics from code
    Returns a feature vector
    """
    import ast
    
    metrics = {}
    
    try:
        tree = ast.parse(code_text)
        
        # Lines of Code
        metrics['loc'] = len(code_text.split('\n'))
        
        # Count different node types
        metrics['num_classes'] = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        metrics['num_functions'] = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        metrics['num_imports'] = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.Import, ast.ImportFrom)))
        metrics['num_loops'] = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.For, ast.While)))
        metrics['num_conditionals'] = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.If))
        metrics['num_assignments'] = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Assign))
        
        # Cyclomatic complexity (simplified)
        complexity = 1  # Start with 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        metrics['cyclomatic_complexity'] = complexity
        
        # Nesting depth
        def get_max_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                child_depth = get_max_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth
        
        metrics['max_nesting_depth'] = get_max_depth(tree)
        
        # Count operators
        metrics['num_operators'] = sum(1 for _ in ast.walk(tree) 
                                      if isinstance(_, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                                       ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr)))
        
    except Exception as e:
        # Default metrics if parsing fails
        metrics = {
            'loc': len(code_text.split('\n')),
            'num_classes': 0,
            'num_functions': 0,
            'num_imports': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'num_assignments': 0,
            'cyclomatic_complexity': 1,
            'max_nesting_depth': 0,
            'num_operators': 0
        }
    
    return metrics


def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic dataset for demonstration"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, 30)
    
    # Create non-linear decision boundary
    defect_score = (X[:, 0] ** 2 + X[:, 1] ** 2 + 
                   0.5 * X[:, 2] - 0.3 * X[:, 3] +
                   np.random.randn(n_samples) * 0.5)
    
    # Create imbalanced labels (20% defective)
    threshold = np.percentile(defect_score, 80)
    y = (defect_score > threshold).astype(int)
    
    return X, y


if __name__ == "__main__":
    # Generate demo data
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples=1000)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining defect prediction model...")
    predictor = DefectPredictor(n_features=15)
    predictor.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")