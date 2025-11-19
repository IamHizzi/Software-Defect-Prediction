"""
Enhanced Model Training Pipeline for 90%+ Accuracy
Uses advanced ensemble techniques and hyperparameter optimization
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
warnings.filterwarnings('ignore')

from nasa_dataset_loader import NASADatasetLoader


class EnhancedDefectPredictor:
    """
    Enhanced defect predictor with advanced ensemble methods
    Target: 90%+ accuracy
    """

    def __init__(self, n_features=15, use_advanced_ensemble=True):
        """
        Initialize enhanced predictor

        Args:
            n_features: Number of features to select
            use_advanced_ensemble: Use stacking ensemble (better accuracy)
        """
        self.n_features = n_features
        self.use_advanced_ensemble = use_advanced_ensemble
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.sampler = SMOTEENN(random_state=42)  # Better than SMOTETomek
        self.ensemble = None
        self.selected_features = None

    def build_advanced_ensemble(self):
        """Build stacking ensemble with multiple base models"""

        # Base models - diverse set
        base_models = [
            ('rf1', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('rf2', RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                max_features='log2',
                random_state=43,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=4,
                subsample=0.8,
                random_state=42
            )),
            ('ada', AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )),
            ('svm', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ]

        # Meta-learner
        meta_learner = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )

        # Stacking ensemble
        self.ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

    def build_voting_ensemble(self):
        """Build voting ensemble (faster alternative)"""

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        svm = SVC(
            kernel='rbf',
            C=10,
            probability=True,
            random_state=42
        )

        self.ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft',
            weights=[2, 2, 1],
            n_jobs=-1
        )

    def select_features(self, X, y):
        """Advanced feature selection"""

        # Use mutual information
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(self.n_features, X.shape[1])
        )
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = self.feature_selector.get_support(indices=True)

        return X_selected

    def balance_classes(self, X, y):
        """Advanced class balancing with SMOTE-ENN"""
        try:
            X_balanced, y_balanced = self.sampler.fit_resample(X, y)
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Warning: Could not apply SMOTE-ENN: {e}")
            # Fallback to SMOTE only
            try:
                smote = SMOTE(random_state=42)
                return smote.fit_resample(X, y)
            except:
                return X, y

    def train(self, X, y, optimize_hyperparameters=False):
        """
        Train enhanced model

        Args:
            X: Features
            y: Labels
            optimize_hyperparameters: Whether to run grid search (slower)
        """
        print(f"Training enhanced model...")
        print(f"  Input shape: {X.shape}")
        print(f"  Class distribution: {np.bincount(y)}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"  Features scaled")

        # Feature selection
        X_selected = self.select_features(X_scaled, y)
        print(f"  Features selected: {X_selected.shape[1]}")

        # Balance classes
        X_balanced, y_balanced = self.balance_classes(X_selected, y)
        print(f"  Data balanced: {X_balanced.shape[0]} samples")
        print(f"  Balanced distribution: {np.bincount(y_balanced)}")

        # Build ensemble
        if self.use_advanced_ensemble:
            print(f"  Building stacking ensemble...")
            self.build_advanced_ensemble()
        else:
            print(f"  Building voting ensemble...")
            self.build_voting_ensemble()

        # Train
        print(f"  Training ensemble...")
        self.ensemble.fit(X_balanced, y_balanced)
        print(f"  ✓ Training complete")

        return self

    def predict(self, X):
        """Predict defect probability"""
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        predictions = self.ensemble.predict(X_selected)
        probabilities = self.ensemble.predict_proba(X_selected)
        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        predictions, probabilities = self.predict(X_test)

        # Calculate all metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        # ROC-AUC (if binary classification)
        try:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        except:
            roc_auc = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }

        return results


class EnhancedTrainingPipeline:
    """Enhanced training pipeline for NASA datasets"""

    def __init__(self, output_dir='./enhanced_models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def train_dataset(self, dataset_name, X, y, test_size=0.2):
        """Train on single dataset with enhanced methods"""

        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING: {dataset_name}")
        print('='*70)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nData Split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Train defect rate: {np.mean(y_train)*100:.1f}%")
        print(f"  Test defect rate:  {np.mean(y_test)*100:.1f}%")

        # Train enhanced model
        predictor = EnhancedDefectPredictor(
            n_features=min(15, X.shape[1]),
            use_advanced_ensemble=True
        )
        predictor.train(X_train, y_train)

        # Evaluate
        print(f"\n{'='*70}")
        print(f"EVALUATION")
        print('='*70)

        results = predictor.evaluate(X_test, y_test)

        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  TN: {results['confusion_matrix'][0,0]:<6} FP: {results['confusion_matrix'][0,1]}")
        print(f"  FN: {results['confusion_matrix'][1,0]:<6} TP: {results['confusion_matrix'][1,1]}")

        # Cross-validation
        print(f"\nCross-Validation (5-fold):")
        X_scaled = predictor.scaler.transform(X)
        X_selected = predictor.feature_selector.transform(X_scaled)

        cv_scores = cross_val_score(
            predictor.ensemble, X_selected, y,
            cv=5, scoring='f1', n_jobs=-1
        )

        print(f"  F1 Scores: {cv_scores}")
        print(f"  Mean F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Save model
        model_path = os.path.join(self.models_dir, f'{dataset_name}_enhanced.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        print(f"\n✓ Model saved: {model_path}")

        # Prepare results
        results_summary = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Enhanced Stacking Ensemble',
            'metrics': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'roc_auc': float(results['roc_auc'])
            },
            'cross_validation': {
                'scores': cv_scores.tolist(),
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std())
            },
            'confusion_matrix': results['confusion_matrix'].tolist()
        }

        return {
            'model': predictor,
            'results': results_summary,
            'X_test': X_test,
            'y_test': y_test
        }

    def train_all_datasets(self, datasets):
        """Train enhanced models on all datasets"""

        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING PIPELINE")
        print(f"Training on {len(datasets)} NASA datasets")
        print('='*70)

        all_results = {}

        for dataset_name, (X, y, info) in datasets.items():
            try:
                result = self.train_dataset(dataset_name, X, y)
                all_results[dataset_name] = result
            except Exception as e:
                print(f"\n❌ Error training {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_results

    def save_results(self, all_results):
        """Save comprehensive results"""

        # JSON results
        import json

        results_json = {}
        for name, result in all_results.items():
            results_json[name] = result['results']

        json_path = os.path.join(self.results_dir, 'enhanced_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n✓ Results saved: {json_path}")

        # CSV summary
        summary_data = []
        for name, result in all_results.items():
            m = result['results']['metrics']
            cv = result['results']['cross_validation']

            summary_data.append({
                'Dataset': name,
                'Accuracy': f"{m['accuracy']*100:.2f}%",
                'Precision': f"{m['precision']:.4f}",
                'Recall': f"{m['recall']:.4f}",
                'F1-Score': f"{m['f1_score']:.4f}",
                'ROC-AUC': f"{m['roc_auc']:.4f}",
                'CV F1 Mean': f"{cv['mean']:.4f}",
                'CV F1 Std': f"{cv['std']:.4f}"
            })

        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, 'enhanced_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary saved: {csv_path}")

        # Text report
        report = self.generate_report(all_results)
        report_path = os.path.join(self.results_dir, 'enhanced_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved: {report_path}")

        return df

    def generate_report(self, all_results):
        """Generate comprehensive report"""

        lines = []
        lines.append("="*70)
        lines.append("ENHANCED NASA DEFECT PREDICTION - TRAINING REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model Type: Advanced Stacking Ensemble")
        lines.append(f"Datasets Trained: {len(all_results)}")
        lines.append("")

        # Overall statistics
        accuracies = [r['results']['metrics']['accuracy'] for r in all_results.values()]
        f1_scores = [r['results']['metrics']['f1_score'] for r in all_results.values()]

        lines.append("OVERALL PERFORMANCE")
        lines.append("-"*70)
        lines.append(f"Average Accuracy:  {np.mean(accuracies)*100:.2f}%")
        lines.append(f"Average F1-Score:  {np.mean(f1_scores):.4f}")
        lines.append(f"Best Accuracy:     {np.max(accuracies)*100:.2f}%")
        lines.append(f"Best F1-Score:     {np.max(f1_scores):.4f}")
        lines.append("")

        # Individual results
        lines.append("DETAILED RESULTS BY DATASET")
        lines.append("-"*70)

        for name, result in all_results.items():
            m = result['results']['metrics']
            cv = result['results']['cross_validation']

            lines.append(f"\n{name}:")
            lines.append(f"  Accuracy:  {m['accuracy']*100:.2f}%")
            lines.append(f"  Precision: {m['precision']:.4f}")
            lines.append(f"  Recall:    {m['recall']:.4f}")
            lines.append(f"  F1-Score:  {m['f1_score']:.4f}")
            lines.append(f"  ROC-AUC:   {m['roc_auc']:.4f}")
            lines.append(f"  CV F1:     {cv['mean']:.4f} (+/- {cv['std']:.4f})")

        lines.append("\n" + "="*70)

        return "\n".join(lines)


def main():
    """Main enhanced training pipeline"""

    print("\n" + "="*70)
    print("ENHANCED MODEL TRAINING FOR 90%+ ACCURACY")
    print("="*70)

    # Load datasets
    print("\nStep 1: Loading NASA datasets...")
    loader = NASADatasetLoader()
    datasets = loader.load_all_datasets(['CM1', 'JM1', 'KC1', 'KC2', 'PC1'])

    if not datasets:
        print("❌ No datasets loaded")
        return

    # Train enhanced models
    print("\nStep 2: Training enhanced models...")
    pipeline = EnhancedTrainingPipeline()
    results = pipeline.train_all_datasets(datasets)

    if not results:
        print("❌ No models trained")
        return

    # Save results
    print("\nStep 3: Saving results...")
    summary = pipeline.save_results(results)

    # Display summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print('='*70)
    print(summary.to_string(index=False))

    print(f"\n{'='*70}")
    print("✓ ENHANCED TRAINING COMPLETE")
    print('='*70)
    print(f"Models saved in: {pipeline.models_dir}")
    print(f"Results saved in: {pipeline.results_dir}")


if __name__ == "__main__":
    main()
