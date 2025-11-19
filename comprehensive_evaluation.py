#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Unified Defect Mitigation Framework
Evaluates all three phases with thesis-specified metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from phase1_prediction import DefectPredictor
from phase2_localization import DefectLocalizer
from phase3_bug_fix import BugFixGenerator
from nasa_dataset_loader import NASADatasetLoader
import json
from datetime import datetime

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def evaluate_phase1(dataset_name='JM1'):
    """
    Evaluate Phase 1: Defect Prediction
    Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    Targets: F1≥0.85, AUC≥0.85, Accuracy≥0.85
    """
    print_section("PHASE 1 EVALUATION: DEFECT PREDICTION")

    # Load dataset
    loader = NASADatasetLoader()
    X, y, info = loader.load_dataset(dataset_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    predictor = DefectPredictor()
    predictor.train(X_train, y_train)

    # Predict
    y_pred, y_proba = predictor.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Display results
    print(f"Dataset: {dataset_name}")
    print(f"Test Size: {len(y_test)} samples")
    print(f"\nPerformance Metrics:")
    print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

    # Thesis targets
    print("\n" + "-"*80)
    print("THESIS TARGET EVALUATION:")
    print("-"*80)
    print(f"  Accuracy ≥ 0.85:  {accuracy:.4f} {'✓ PASS' if accuracy >= 0.85 else '✗ FAIL'}")
    print(f"  F1-Score ≥ 0.85:  {f1:.4f} {'✓ PASS' if f1 >= 0.85 else '✗ FAIL'}")
    print(f"  ROC-AUC ≥ 0.85:   {roc_auc:.4f} {'✓ PASS' if roc_auc >= 0.85 else '✗ FAIL'}")

    results = {
        'phase': 'Phase 1: Defect Prediction',
        'dataset': dataset_name,
        'test_size': len(y_test),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'thesis_targets': {
            'accuracy': {'target': 0.85, 'achieved': float(accuracy), 'pass': accuracy >= 0.85},
            'f1_score': {'target': 0.85, 'achieved': float(f1), 'pass': f1 >= 0.85},
            'roc_auc': {'target': 0.85, 'achieved': float(roc_auc), 'pass': roc_auc >= 0.85}
        }
    }

    return results

def evaluate_phase2():
    """
    Evaluate Phase 2: Defect Localization
    Metrics: Top-3 Accuracy, Precision@K, Recall@K
    Target: Top-3 accuracy ≥ 70%
    """
    print_section("PHASE 2 EVALUATION: DEFECT LOCALIZATION")

    # Test cases with known buggy lines
    test_cases = [
        {
            'name': 'Division by Zero',
            'code': '''
def calculate_average(numbers):
    total = sum(numbers)
    avg = total / len(numbers)  # Line 3: Bug here
    return avg
''',
            'buggy_lines': [3]
        },
        {
            'name': 'Index Error',
            'code': '''
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result[0]  # Line 5: Bug here
''',
            'buggy_lines': [5]
        },
        {
            'name': 'None Type Error',
            'code': '''
class Processor:
    def __init__(self):
        self.data = None
    def process(self):
        return len(self.data)  # Line 5: Bug here
''',
            'buggy_lines': [5]
        }
    ]

    localizer = DefectLocalizer()

    total_cases = len(test_cases)
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    print(f"Evaluating {total_cases} test cases...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"  Known buggy lines: {test_case['buggy_lines']}")

        try:
            results = localizer.localize_defects(test_case['code'], defect_prob=0.8, top_n=5)
            predicted_lines = results['top_lines']

            print(f"  Predicted lines: {predicted_lines}")

            # Check if any buggy line is in top-K predictions
            buggy_set = set(test_case['buggy_lines'])

            if predicted_lines:
                if buggy_set & set(predicted_lines[:1]):
                    top1_correct += 1
                    print(f"  ✓ Top-1: Correct")
                if buggy_set & set(predicted_lines[:3]):
                    top3_correct += 1
                    print(f"  ✓ Top-3: Correct")
                if buggy_set & set(predicted_lines[:5]):
                    top5_correct += 1
                    print(f"  ✓ Top-5: Correct")
            else:
                print(f"  ✗ No predictions")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    # Calculate accuracies
    top1_acc = top1_correct / total_cases
    top3_acc = top3_correct / total_cases
    top5_acc = top5_correct / total_cases

    print("-"*80)
    print("LOCALIZATION ACCURACY:")
    print("-"*80)
    print(f"  Top-1 Accuracy: {top1_acc:.2%} ({top1_correct}/{total_cases})")
    print(f"  Top-3 Accuracy: {top3_acc:.2%} ({top3_correct}/{total_cases})")
    print(f"  Top-5 Accuracy: {top5_acc:.2%} ({top5_correct}/{total_cases})")

    # Thesis target
    print("\n" + "-"*80)
    print("THESIS TARGET EVALUATION:")
    print("-"*80)
    print(f"  Top-3 Accuracy ≥ 70%:  {top3_acc:.2%} {'✓ PASS' if top3_acc >= 0.70 else '✗ FAIL'}")

    results = {
        'phase': 'Phase 2: Defect Localization',
        'test_cases': total_cases,
        'metrics': {
            'top1_accuracy': float(top1_acc),
            'top3_accuracy': float(top3_acc),
            'top5_accuracy': float(top5_acc),
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'top5_correct': top5_correct
        },
        'thesis_targets': {
            'top3_accuracy': {'target': 0.70, 'achieved': float(top3_acc), 'pass': top3_acc >= 0.70}
        }
    }

    return results

def evaluate_phase3():
    """
    Evaluate Phase 3: Bug Fix Generation
    Metrics: Valid Fix Rate, Syntax Correctness
    Target: Valid fix rate ≥ 80%
    """
    print_section("PHASE 3 EVALUATION: BUG FIX GENERATION (RATG)")

    # Test cases with buggy code
    test_cases = [
        {
            'name': 'Division by Zero',
            'code': 'result = total / count',
            'suspicious_lines': [1]
        },
        {
            'name': 'Index Out of Range',
            'code': 'item = data[index + 1]',
            'suspicious_lines': [1]
        },
        {
            'name': 'None Type Error',
            'code': 'length = len(data.items)',
            'suspicious_lines': [1]
        },
        {
            'name': 'Bare Except',
            'code': 'try:\n    process()\nexcept:\n    pass',
            'suspicious_lines': [3]
        }
    ]

    generator = BugFixGenerator()

    total_cases = len(test_cases)
    valid_fixes = 0
    syntax_valid = 0

    print(f"Evaluating {total_cases} test cases...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"  Buggy code: {test_case['code'][:50]}...")

        try:
            fixed_code, applied_fixes = generator.generate_fix(
                test_case['code'],
                test_case['suspicious_lines']
            )

            # Check syntax validity
            import ast
            try:
                ast.parse(fixed_code)
                syntax_valid += 1
                print(f"  ✓ Syntax valid")
            except:
                print(f"  ✗ Syntax invalid")

            # Check if fixes were applied
            if applied_fixes:
                valid_fixes += 1
                print(f"  ✓ Fixes applied: {len(applied_fixes)}")
            else:
                print(f"  - No fixes applied")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    # Calculate rates
    valid_fix_rate = valid_fixes / total_cases
    syntax_validity_rate = syntax_valid / total_cases

    print("-"*80)
    print("BUG FIX GENERATION METRICS:")
    print("-"*80)
    print(f"  Valid Fix Rate:        {valid_fix_rate:.2%} ({valid_fixes}/{total_cases})")
    print(f"  Syntax Validity Rate:  {syntax_validity_rate:.2%} ({syntax_valid}/{total_cases})")

    # Thesis target
    print("\n" + "-"*80)
    print("THESIS TARGET EVALUATION:")
    print("-"*80)
    print(f"  Valid Fix Rate ≥ 80%:  {valid_fix_rate:.2%} {'✓ PASS' if valid_fix_rate >= 0.80 else '✗ FAIL'}")

    results = {
        'phase': 'Phase 3: Bug Fix Generation (RATG)',
        'test_cases': total_cases,
        'metrics': {
            'valid_fix_rate': float(valid_fix_rate),
            'syntax_validity_rate': float(syntax_validity_rate),
            'valid_fixes': valid_fixes,
            'syntax_valid': syntax_valid
        },
        'thesis_targets': {
            'valid_fix_rate': {'target': 0.80, 'achieved': float(valid_fix_rate), 'pass': valid_fix_rate >= 0.80}
        }
    }

    return results

def main():
    print("="*80)
    print("COMPREHENSIVE FRAMEWORK EVALUATION")
    print("="*80)
    print(f"Evaluation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Evaluate Phase 1
    try:
        phase1_results = evaluate_phase1()
        all_results['phase1'] = phase1_results
    except Exception as e:
        print(f"Error in Phase 1 evaluation: {e}")
        all_results['phase1'] = {'error': str(e)}

    # Evaluate Phase 2
    try:
        phase2_results = evaluate_phase2()
        all_results['phase2'] = phase2_results
    except Exception as e:
        print(f"Error in Phase 2 evaluation: {e}")
        all_results['phase2'] = {'error': str(e)}

    # Evaluate Phase 3
    try:
        phase3_results = evaluate_phase3()
        all_results['phase3'] = phase3_results
    except Exception as e:
        print(f"Error in Phase 3 evaluation: {e}")
        all_results['phase3'] = {'error': str(e)}

    # Overall summary
    print_section("OVERALL EVALUATION SUMMARY")

    print("Phase 1 - Defect Prediction:")
    if 'metrics' in all_results.get('phase1', {}):
        p1 = all_results['phase1']['metrics']
        print(f"  Accuracy: {p1['accuracy']:.4f}")
        print(f"  F1-Score: {p1['f1_score']:.4f}")
        print(f"  ROC-AUC:  {p1['roc_auc']:.4f}")

    print("\nPhase 2 - Defect Localization:")
    if 'metrics' in all_results.get('phase2', {}):
        p2 = all_results['phase2']['metrics']
        print(f"  Top-3 Accuracy: {p2['top3_accuracy']:.2%}")

    print("\nPhase 3 - Bug Fix Generation:")
    if 'metrics' in all_results.get('phase3', {}):
        p3 = all_results['phase3']['metrics']
        print(f"  Valid Fix Rate: {p3['valid_fix_rate']:.2%}")

    # Save results
    all_results['timestamp'] = datetime.now().isoformat()

    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("Results saved to: evaluation_results.json")
    print("="*80)

    return all_results

if __name__ == "__main__":
    results = main()
