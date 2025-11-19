"""
PHASE 1: Defect Prediction using Ensemble Machine Learning
Target: F1-score ≥ 85%, AUC ≥ 0.85, Accuracy ≥ 85%

Based on thesis proposal:
- Ensemble Voting Classifier (Random Forest + SVM + Decision Tree)
- SMOTE-TOMEK for class imbalance
- MIC and Correlation for feature selection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DefectPredictor:
    """
    Phase 1: ML-based Defect Prediction
    Uses ensemble of RF, SVM, and Decision Tree
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.sampler = SMOTETomek(random_state=42)
        self.ensemble = None
        self.selected_features_idx = None

    def feature_selection_mic(self, X, y, k=15):
        """
        Feature Selection using Mutual Information
        (Simplified alternative to MIC for better compatibility)
        """
        print("\n" + "="*70)
        print("FEATURE SELECTION - Mutual Information")
        print("="*70)

        # Use mutual information for feature selection
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature indices
        selected_idx = selector.get_support(indices=True)
        scores = selector.scores_

        print(f"Total features: {X.shape[1]}")
        print(f"Selected features: {len(selected_idx)}")
        print(f"Top 5 MI scores: {sorted(scores, reverse=True)[:5]}")

        self.selected_features_idx = selected_idx
        return X_selected

    def apply_smote_tomek(self, X, y):
        """
        Address Class Imbalance using SMOTE-TOMEK
        """
        print("\n" + "="*70)
        print("CLASS BALANCING - SMOTE-TOMEK")
        print("="*70)

        print(f"Before balancing:")
        print(f"  Total samples: {len(y)}")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(y)*100:.1f}%)")

        X_balanced, y_balanced = self.sampler.fit_resample(X, y)

        print(f"\nAfter balancing:")
        print(f"  Total samples: {len(y_balanced)}")
        unique, counts = np.unique(y_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(y_balanced)*100:.1f}%)")

        return X_balanced, y_balanced

    def build_ensemble(self):
        """
        Build Ensemble Voting Classifier
        Base classifiers: Random Forest, SVM, Decision Tree
        """
        print("\n" + "="*70)
        print("BUILDING ENSEMBLE MODEL")
        print("="*70)

        # Base classifiers as specified in thesis
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        svm = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )

        dt = DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        )

        # Voting Classifier
        self.ensemble = VotingClassifier(
            estimators=[
                ('RandomForest', rf),
                ('SVM', svm),
                ('DecisionTree', dt)
            ],
            voting='soft',
            n_jobs=-1
        )

        print("Ensemble created with:")
        print("  - Random Forest (n_estimators=100, max_depth=10)")
        print("  - SVM (kernel=rbf)")
        print("  - Decision Tree (max_depth=8)")
        print("  - Voting: Soft")

    def train(self, X_train, y_train):
        """
        Train the ensemble model
        """
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"Features scaled using StandardScaler")

        # Feature selection
        X_train_selected = self.feature_selection_mic(X_train_scaled, y_train)

        # Balance classes
        X_train_balanced, y_train_balanced = self.apply_smote_tomek(
            X_train_selected, y_train
        )

        # Build ensemble
        self.build_ensemble()

        # Train
        print("\nTraining ensemble model...")
        self.ensemble.fit(X_train_balanced, y_train_balanced)
        print("✓ Training complete!")

        return self

    def predict(self, X):
        """
        Make predictions
        """
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features_idx]
        predictions = self.ensemble.predict(X_selected)
        probabilities = self.ensemble.predict_proba(X_selected)
        return predictions, probabilities

    def evaluate(self, X_test, y_test, save_plots=True):
        """
        Evaluate model performance
        Target: F1-score ≥ 85%, AUC ≥ 0.85, Accuracy ≥ 85%
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

        predictions, probabilities = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, probabilities[:, 1])
        except:
            auc = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Print results
        print(f"\nPERFORMANCE METRICS:")
        print(f"{'─'*70}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"AUC:       {auc:.4f}")
        print(f"{'─'*70}")

        # Check targets
        print(f"\nTARGET VALIDATION:")
        print(f"  Accuracy ≥ 85%:  {'✓' if accuracy >= 0.85 else '✗'} ({accuracy*100:.2f}%)")
        print(f"  F1-Score ≥ 85%:  {'✓' if f1 >= 0.85 else '✗'} ({f1*100:.2f}%)")
        print(f"  AUC ≥ 0.85:      {'✓' if auc >= 0.85 else '✗'} ({auc:.4f})")

        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"  TN: {cm[0,0]:<6} FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]}")

        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Plot confusion matrix
        if save_plots:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Phase 1: Defect Prediction')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig('phase1_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix saved: phase1_confusion_matrix.png")
            plt.close()

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }

        return results


def demo_phase1():
    """
    Demonstration of Phase 1: Defect Prediction
    """
    print("\n" + "="*70)
    print("PHASE 1: DEFECT PREDICTION - DEMONSTRATION")
    print("="*70)

    # Load NASA dataset
    from nasa_dataset_loader import NASADatasetLoader

    print("\nSTEP 1: Loading Dataset...")
    loader = NASADatasetLoader()
    X, y, info = loader.load_dataset('JM1')

    print(f"\nDataset loaded: {info['name']}")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Defective: {info['n_defective']} ({info['defect_rate']*100:.1f}%)")

    # Split data
    print("\nSTEP 2: Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Train model
    print("\nSTEP 3: Training Model...")
    predictor = DefectPredictor()
    predictor.train(X_train, y_train)

    # Evaluate
    print("\nSTEP 4: Evaluating Model...")
    results = predictor.evaluate(X_test, y_test, save_plots=True)

    print("\n" + "="*70)
    print("✓ PHASE 1 COMPLETE")
    print("="*70)

    return predictor, results


if __name__ == "__main__":
    predictor, results = demo_phase1()
