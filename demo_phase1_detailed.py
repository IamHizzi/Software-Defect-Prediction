#!/usr/bin/env python3
"""
Detailed Demo for Phase 1: Defect Prediction
Captures outputs for each implementation step
"""

import numpy as np
import pandas as pd
from phase1_prediction import DefectPredictor
from nasa_dataset_loader import NASADatasetLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print("="*80)
    print("PHASE 1: DEFECT PREDICTION - DETAILED DEMONSTRATION")
    print("="*80)

    # Step 1: Load Dataset
    print_section("STEP 1: LOADING NASA PROMISE DATASET")

    loader = NASADatasetLoader()
    print("Available datasets: CM1, JM1, KC1, KC2, PC1")
    print(f"Loading dataset: JM1 (largest dataset)")

    X, y, dataset_info = loader.load_dataset('JM1')

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Total samples: {X.shape[0]:,}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Defective samples: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  - Non-defective samples: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  - Class imbalance ratio: 1:{np.sum(y == 0)/np.sum(y == 1):.1f}")

    # Display first few rows
    print("\nFirst 5 samples (features):")
    df = pd.DataFrame(X[:5])
    print(df.to_string(index=False))

    print("\nLabels for first 10 samples:")
    print(f"  {y[:10]} (0=Non-defective, 1=Defective)")

    # Step 2: Initialize Predictor
    print_section("STEP 2: INITIALIZING DEFECT PREDICTOR")

    predictor = DefectPredictor()
    print("✓ DefectPredictor initialized")
    print("\nComponents:")
    print("  - Feature Selector: Mutual Information (k=15)")
    print("  - Sampler: SMOTE-TOMEK (hybrid resampling)")
    print("  - Scaler: StandardScaler")
    print("  - Ensemble: Voting Classifier")
    print("    • Random Forest (n_estimators=200)")
    print("    • SVM (kernel=rbf, C=10)")
    print("    • Decision Tree (max_depth=10)")

    # Step 3: Feature Selection
    print_section("STEP 3: FEATURE SELECTION (Mutual Information)")

    print(f"Original features: {X.shape[1]}")
    X_selected = predictor.feature_selection_mic(X, y, k=15)
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"✓ Reduced dimensionality by {X.shape[1] - X_selected.shape[1]} features")

    # Show feature importance
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(X, y)
    top_features_idx = np.argsort(mi_scores)[-15:][::-1]

    print("\nTop 15 features by Mutual Information:")
    for i, idx in enumerate(top_features_idx, 1):
        print(f"  {i:2d}. Feature {idx:2d}: MI Score = {mi_scores[idx]:.4f}")

    # Step 4: Data Balancing (SMOTE-TOMEK)
    print_section("STEP 4: DATA BALANCING (SMOTE-TOMEK)")

    print(f"Before balancing:")
    print(f"  - Class 0 (Non-defective): {np.sum(y == 0):,} samples")
    print(f"  - Class 1 (Defective): {np.sum(y == 1):,} samples")
    print(f"  - Imbalance ratio: 1:{np.sum(y == 0)/np.sum(y == 1):.2f}")

    X_balanced, y_balanced = predictor.apply_smote_tomek(X_selected, y)

    print(f"\nAfter SMOTE-TOMEK balancing:")
    print(f"  - Class 0 (Non-defective): {np.sum(y_balanced == 0):,} samples")
    print(f"  - Class 1 (Defective): {np.sum(y_balanced == 1):,} samples")
    print(f"  - Imbalance ratio: 1:{np.sum(y_balanced == 0)/np.sum(y_balanced == 1):.2f}")
    print(f"  - Total samples: {len(y_balanced):,} (from {len(y):,})")

    print("\n✓ SMOTE: Oversampled minority class")
    print("✓ TOMEK: Removed noisy boundary samples")

    # Step 5: Train Ensemble Model
    print_section("STEP 5: TRAINING ENSEMBLE MODEL")

    print("Training with train-test split (80-20)...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    predictor.train(X_train, y_train)

    print("\n✓ Training completed!")
    print("\nModel Architecture:")
    print("┌─────────────────────────────────────────┐")
    print("│         VOTING CLASSIFIER               │")
    print("│         (Soft Voting)                   │")
    print("├─────────────────────────────────────────┤")
    print("│  1. Random Forest                       │")
    print("│     - n_estimators: 200                 │")
    print("│     - max_depth: 15                     │")
    print("│     - min_samples_split: 2              │")
    print("├─────────────────────────────────────────┤")
    print("│  2. Support Vector Machine              │")
    print("│     - kernel: rbf                       │")
    print("│     - C: 10                             │")
    print("│     - probability: True                 │")
    print("├─────────────────────────────────────────┤")
    print("│  3. Decision Tree                       │")
    print("│     - max_depth: 10                     │")
    print("│     - min_samples_split: 2              │")
    print("└─────────────────────────────────────────┘")

    # Step 6: Evaluation Metrics
    print_section("STEP 6: EVALUATION METRICS")

    print("Evaluating model on test set...")
    y_pred, y_proba = predictor.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    # Cross validation on training set
    cv_scores = cross_val_score(predictor.ensemble,
                                 predictor.scaler.transform(X_train),
                                 y_train,
                                 cv=5,
                                 scoring='f1')

    print("\nPerformance Metrics:")
    print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc:.4f}")

    print(f"\nCross-Validation Results (5-fold):")
    print(f"  • Mean F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  • CV Scores: {cv_scores}")

    # Target metrics from thesis
    print("\n" + "-"*80)
    print("THESIS TARGET METRICS:")
    print("-"*80)
    print(f"  Target: F1-Score ≥ 0.85    | Achieved: {f1:.4f} {'✓' if f1 >= 0.85 else '✗'}")
    print(f"  Target: ROC-AUC ≥ 0.85     | Achieved: {roc_auc:.4f} {'✓' if roc_auc >= 0.85 else '✗'}")
    print(f"  Target: Accuracy ≥ 0.85    | Achieved: {accuracy:.4f} {'✓' if accuracy >= 0.85 else '✗'}")

    # Store results for later use
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_proba': y_proba
    }

    # Step 7: Confusion Matrix
    print_section("STEP 7: CONFUSION MATRIX")

    y_pred = results['y_pred']
    y_true = results['y_test']
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print("                Predicted")
    print("              Non-Def  Defective")
    print(f"Actual Non-Def   {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"       Defective {cm[1,0]:4d}     {cm[1,1]:4d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nDetailed Breakdown:")
    print(f"  • True Negatives (TN):  {tn:4d} - Correctly predicted non-defective")
    print(f"  • False Positives (FP): {fp:4d} - Incorrectly predicted as defective")
    print(f"  • False Negatives (FN): {fn:4d} - Missed defective modules")
    print(f"  • True Positives (TP):  {tp:4d} - Correctly predicted defective")

    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Defective', 'Defective'],
                yticklabels=['Non-Defective', 'Defective'])
    plt.title('Confusion Matrix - Phase 1: Defect Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('phase1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✓ Confusion matrix saved: phase1_confusion_matrix.png")
    plt.close()

    # Step 8: ROC Curve
    print_section("STEP 8: ROC CURVE")

    y_proba = results['y_proba'][:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Phase 1: Defect Prediction')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase1_roc_curve.png', dpi=150, bbox_inches='tight')
    print(f"ROC-AUC Score: {roc_auc_score:.4f}")
    print("✓ ROC curve saved: phase1_roc_curve.png")
    plt.close()

    # Step 9: Feature Importance
    print_section("STEP 9: FEATURE IMPORTANCE (Random Forest)")

    # Get Random Forest from ensemble
    rf_estimator = predictor.ensemble.named_estimators_['RandomForest']
    feature_importance = rf_estimator.feature_importances_

    print("Top 10 Most Important Features:")
    indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. Feature {idx:2d}: {feature_importance[idx]:.4f}")

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('phase1_feature_importance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Feature importance chart saved: phase1_feature_importance.png")
    plt.close()

    # Summary
    print_section("PHASE 1 SUMMARY")

    print("✓ All steps completed successfully!")
    print("\nGenerated Outputs:")
    print("  1. phase1_confusion_matrix.png")
    print("  2. phase1_roc_curve.png")
    print("  3. phase1_feature_importance.png")

    print("\nKey Results:")
    print(f"  • Dataset: JM1 ({X.shape[0]:,} samples)")
    print(f"  • Features: {X_selected.shape[1]} (selected from {X.shape[1]})")
    print(f"  • Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  • F1-Score: {results['f1']:.4f}")
    print(f"  • ROC-AUC: {results['roc_auc']:.4f}")

    print("\n" + "="*80)
    print("PHASE 1 DEMONSTRATION COMPLETE")
    print("="*80)

    return results

if __name__ == "__main__":
    results = main()
