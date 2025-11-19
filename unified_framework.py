"""
Unified ML-Graph Framework for Automated Software Defect Mitigation

Main framework integrating:
1. Defect Prediction (Ensemble ML)
2. Defect Localization (GAT on ASTs)
3. Bug Fix Generation (RATG)
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import modules
from defect_prediction import DefectPredictor, extract_software_metrics
from defect_localization import DefectLocalizer
from bug_fix import BugFixer


class UnifiedDefectMitigationFramework:
    """
    End-to-end framework for automated software defect mitigation
    """
    
    def __init__(self):
        self.predictor = DefectPredictor(n_features=15)
        self.localizer = DefectLocalizer(hidden_channels=64, num_heads=4)
        self.fixer = BugFixer()
        self.is_trained = False
        
    def train_predictor(self, X_train, y_train):
        """Train the defect prediction module"""
        print("Training defect prediction module...")
        self.predictor.train(X_train, y_train)
        self.is_trained = True
        print("✓ Defect predictor trained")
        
    def train_localizer(self, train_data, epochs=30):
        """Train the defect localization module"""
        print("Training defect localization module...")
        self.localizer.train_model(train_data, epochs=epochs)
        print("✓ Defect localizer trained")
        
    def process_code_file(self, code_text: str, file_path: str = None) -> Dict:
        """
        Process a code file through the entire pipeline
        
        Args:
            code_text: Source code as string
            file_path: Optional file path for reference
            
        Returns:
            Dictionary with results from all phases
        """
        results = {
            'file_path': file_path,
            'original_code': code_text,
            'phase1_prediction': None,
            'phase2_localization': None,
            'phase3_fix': None,
            'is_defective': False,
            'fixed_code': None
        }
        
        print(f"\n{'='*70}")
        print(f"Processing: {file_path or 'Code snippet'}")
        print('='*70)
        
        # Phase 1: Defect Prediction
        print("\n[Phase 1] Defect Prediction...")
        metrics = extract_software_metrics(code_text)
        metrics_vector = np.array(list(metrics.values())).reshape(1, -1)
        
        # Pad metrics if needed
        if metrics_vector.shape[1] < 30:
            padding = np.zeros((1, 30 - metrics_vector.shape[1]))
            metrics_vector = np.hstack([metrics_vector, padding])
        
        if self.is_trained:
            predictions, probabilities = self.predictor.predict(metrics_vector)
            is_defective = predictions[0] == 1
            defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else 0.5
        else:
            # Use heuristic if not trained
            complexity_score = (metrics.get('cyclomatic_complexity', 0) / 10.0 + 
                              metrics.get('max_nesting_depth', 0) / 5.0)
            defect_prob = min(complexity_score, 0.9)
            is_defective = defect_prob > 0.5
        
        results['is_defective'] = is_defective
        results['phase1_prediction'] = {
            'defect_probability': float(defect_prob),
            'metrics': metrics,
            'prediction': 'DEFECTIVE' if is_defective else 'CLEAN'
        }
        
        print(f"  Prediction: {results['phase1_prediction']['prediction']}")
        print(f"  Defect Probability: {defect_prob:.4f}")
        print(f"  Key Metrics: LOC={metrics.get('loc', 0)}, "
              f"Complexity={metrics.get('cyclomatic_complexity', 0)}")
        
        # Phase 2: Defect Localization (only if defective)
        if is_defective:
            print("\n[Phase 2] Defect Localization...")
            localization_results = self.localizer.localize_defects(
                code_text, 
                defect_prob=defect_prob,
                metrics=metrics
            )
            
            defective_lines = self.localizer.get_defective_lines(
                code_text,
                defect_prob=defect_prob,
                metrics=metrics
            )
            
            results['phase2_localization'] = {
                'defective_lines': defective_lines,
                'top_suspicious_nodes': localization_results['top_nodes'][:3],
                'num_flagged_nodes': len(localization_results['defective_nodes'])
            }
            
            print(f"  Located {len(defective_lines)} potentially defective lines")
            if defective_lines:
                print(f"  Line numbers: {defective_lines}")
            
            # Phase 3: Bug Fix Generation
            print("\n[Phase 3] Bug Fix Generation...")
            fixed_code, applied_fixes = self.fixer.generate_fix(
                code_text,
                buggy_lines=defective_lines
            )
            
            results['phase3_fix'] = {
                'fixed_code': fixed_code,
                'applied_fixes': applied_fixes,
                'num_fixes_applied': len(applied_fixes)
            }
            results['fixed_code'] = fixed_code
            
            if applied_fixes:
                print(f"  Applied {len(applied_fixes)} fix(es)")
                for fix in applied_fixes:
                    bug_type = fix.get('type', 'unknown')
                    confidence = fix.get('similarity', 0)
                    print(f"    - {bug_type} (confidence: {confidence:.2f})")
            else:
                print("  No automatic fixes applied (manual review recommended)")
        else:
            print("\n[Phase 2] Skipped (code predicted as clean)")
            print("[Phase 3] Skipped (no defects to fix)")
        
        return results
    
    def batch_process(self, code_files: List[Tuple[str, str]]) -> List[Dict]:
        """
        Process multiple code files
        
        Args:
            code_files: List of (file_path, code_text) tuples
            
        Returns:
            List of results for each file
        """
        results = []
        
        print(f"\nBatch processing {len(code_files)} files...")
        
        for file_path, code_text in code_files:
            result = self.process_code_file(code_text, file_path)
            results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print('='*70)
        
        total_files = len(results)
        defective_files = sum(1 for r in results if r['is_defective'])
        fixed_files = sum(1 for r in results if r['fixed_code'] and r['fixed_code'] != r['original_code'])
        
        print(f"Total files processed: {total_files}")
        print(f"Defective files found: {defective_files}")
        print(f"Files with fixes applied: {fixed_files}")
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = None):
        """Generate detailed report"""
        report = []
        report.append("="*70)
        report.append("DEFECT MITIGATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Phase 1 Results
        report.append("PHASE 1: DEFECT PREDICTION")
        report.append("-" * 70)
        pred = results['phase1_prediction']
        report.append(f"Prediction: {pred['prediction']}")
        report.append(f"Defect Probability: {pred['defect_probability']:.4f}")
        report.append(f"Metrics: {pred['metrics']}")
        report.append("")
        
        # Phase 2 Results
        if results['phase2_localization']:
            report.append("PHASE 2: DEFECT LOCALIZATION")
            report.append("-" * 70)
            loc = results['phase2_localization']
            report.append(f"Defective Lines: {loc['defective_lines']}")
            report.append(f"Flagged Nodes: {loc['num_flagged_nodes']}")
            report.append("")
        
        # Phase 3 Results
        if results['phase3_fix']:
            report.append("PHASE 3: BUG FIX")
            report.append("-" * 70)
            fix = results['phase3_fix']
            report.append(f"Fixes Applied: {fix['num_fixes_applied']}")
            
            if fix['applied_fixes']:
                report.append("\nApplied Fix Details:")
                for f in fix['applied_fixes']:
                    report.append(f"  - Type: {f.get('type', 'unknown')}")
                    report.append(f"    Confidence: {f.get('similarity', 0):.2f}")
            
            if fix['fixed_code']:
                report.append("\nFixed Code:")
                report.append("-" * 70)
                report.append(fix['fixed_code'])
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
        
        return report_text


def demo_unified_framework():
    """Demonstrate the unified framework"""
    
    # Initialize framework
    framework = UnifiedDefectMitigationFramework()
    
    # Example code files
    code_samples = [
        ("example1.py", """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)

def process_items(items):
    results = []
    for i in range(len(items)):
        results.append(items[i+1])
    return results
"""),
        ("example2.py", """
def safe_function(x, y):
    if x is None:
        return 0
    return x * y

def clean_code(data):
    return [item * 2 for item in data if item > 0]
"""),
        ("example3.py", """
def risky_operation(data):
    try:
        result = data['key']
        value = result / 0
        return value
    except:
        pass
    return None
""")
    ]
    
    # Process each file
    print("\n" + "="*70)
    print("UNIFIED DEFECT MITIGATION FRAMEWORK - DEMO")
    print("="*70)
    
    for file_path, code in code_samples:
        results = framework.process_code_file(code, file_path)
        
        # Generate report
        if results['is_defective']:
            print("\n" + "-"*70)
            print("DETAILED REPORT")
            print("-"*70)
            report = framework.generate_report(results)
            print(report)
            print("\n")


if __name__ == "__main__":
    demo_unified_framework()