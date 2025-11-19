"""
MAIN FRAMEWORK: Unified ML-Graph Framework for Automated Software Defect Mitigation

Integrates:
- Phase 1: Defect Prediction (ML-based)
- Phase 2: Defect Localization (GAT-based)
- Phase 3: Bug Fix Recommendation (RATG-based)

Target Performance:
- Phase 1: F1-score ≥ 85%, AUC ≥ 0.85, Accuracy ≥ 85%
- Phase 2: Top-3 localization accuracy ≥ 70%
- Phase 3: Valid fix rate ≥ 80%
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import phases
from phase1_prediction import DefectPredictor
from phase2_localization import DefectLocalizer
from phase3_bug_fix import BugFixGenerator


class UnifiedDefectMitigationFramework:
    """
    Main Framework integrating all three phases
    """

    def __init__(self):
        print("\n" + "="*80)
        print("UNIFIED DEFECT MITIGATION FRAMEWORK")
        print("="*80)
        print("Initializing components...")

        # Initialize phases
        self.phase1 = DefectPredictor()
        self.phase2 = DefectLocalizer(hidden_dim=64, num_heads=4)
        self.phase3 = BugFixGenerator()

        print("✓ Phase 1: Defect Prediction (ML-based)")
        print("✓ Phase 2: Defect Localization (GAT-based)")
        print("✓ Phase 3: Bug Fix Recommendation (RATG-based)")
        print("\n✓ Framework initialized successfully!")

    def process(self, code_files: List[Dict[str, str]]) -> Dict:
        """
        Process code through complete pipeline

        Args:
            code_files: List of dicts with 'path' and 'code' keys

        Returns:
            Complete results from all phases
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE PIPELINE")
        print("="*80)
        print(f"Processing {len(code_files)} files...\n")

        results = {
            'timestamp': datetime.now().isoformat(),
            'files': []
        }

        for file_info in code_files:
            file_path = file_info['path']
            code = file_info['code']

            print(f"\n{'='*80}")
            print(f"Processing: {file_path}")
            print('='*80)

            file_result = self._process_single_file(file_path, code)
            results['files'].append(file_result)

        # Generate summary
        results['summary'] = self._generate_summary(results['files'])

        return results

    def _process_single_file(self, file_path: str, code: str) -> Dict:
        """
        Process a single file through all phases
        """
        result = {
            'file': file_path,
            'original_code': code,
            'phase1': None,
            'phase2': None,
            'phase3': None
        }

        try:
            # PHASE 1: Defect Prediction
            print(f"\n{'─'*80}")
            print("PHASE 1: DEFECT PREDICTION")
            print('─'*80)

            phase1_result = self._run_phase1(code)
            result['phase1'] = phase1_result

            is_defective = phase1_result['is_defective']
            defect_prob = phase1_result['defect_probability']

            print(f"\n  Result: {'⚠️  DEFECTIVE' if is_defective else '✓  CLEAN'}")
            print(f"  Probability: {defect_prob*100:.2f}%")

            # If defective, proceed to Phase 2
            if is_defective:
                # PHASE 2: Defect Localization
                print(f"\n{'─'*80}")
                print("PHASE 2: DEFECT LOCALIZATION")
                print('─'*80)

                phase2_result = self._run_phase2(code, defect_prob)
                result['phase2'] = phase2_result

                suspicious_lines = phase2_result['top_lines']
                print(f"\n  Suspicious lines: {suspicious_lines}")

                # PHASE 3: Bug Fix Generation
                print(f"\n{'─'*80}")
                print("PHASE 3: BUG FIX GENERATION")
                print('─'*80)

                phase3_result = self._run_phase3(code, suspicious_lines)
                result['phase3'] = phase3_result

                print(f"\n  Fixes applied: {len(phase3_result['applied_fixes'])}")

            else:
                print(f"\n  Skipping Phase 2 & 3 (code predicted as clean)")

        except Exception as e:
            print(f"\n❌ Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)

        return result

    def _run_phase1(self, code: str) -> Dict:
        """
        Run Phase 1: Defect Prediction
        Returns: defect probability and prediction
        """
        # Extract software metrics from code
        from defect_prediction import extract_software_metrics

        metrics = extract_software_metrics(code)

        # Prepare feature vector (pad to 21 features for NASA models)
        metrics_vector = np.array(list(metrics.values())).reshape(1, -1)
        if metrics_vector.shape[1] < 21:
            padding = np.zeros((1, 21 - metrics_vector.shape[1]))
            metrics_vector = np.hstack([metrics_vector, padding])
        elif metrics_vector.shape[1] > 21:
            metrics_vector = metrics_vector[:, :21]

        # Make prediction (if model is trained)
        try:
            predictions, probabilities = self.phase1.predict(metrics_vector)
            is_defective = predictions[0] == 1
            defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else 0.5
        except:
            # Use heuristic if model not trained
            complexity = metrics.get('cyclomatic_complexity', 0)
            nesting = metrics.get('max_nesting_depth', 0)
            defect_prob = min((complexity / 10.0 + nesting / 5.0) / 2, 0.95)
            is_defective = defect_prob > 0.5

        return {
            'is_defective': bool(is_defective),
            'defect_probability': float(defect_prob),
            'metrics': metrics
        }

    def _run_phase2(self, code: str, defect_prob: float) -> Dict:
        """
        Run Phase 2: Defect Localization
        Returns: suspicious lines and nodes
        """
        localization_results = self.phase2.localize_defects(
            code,
            defect_prob=defect_prob,
            top_n=3
        )

        return {
            'suspicious_nodes': localization_results['suspicious_nodes'],
            'top_lines': localization_results['top_lines'],
            'scores': localization_results['scores'].tolist() if isinstance(localization_results['scores'], np.ndarray) else localization_results['scores']
        }

    def _run_phase3(self, code: str, suspicious_lines: List[int]) -> Dict:
        """
        Run Phase 3: Bug Fix Generation
        Returns: fixed code and applied fixes
        """
        fixed_code, applied_fixes = self.phase3.generate_fix(
            code,
            suspicious_lines=suspicious_lines
        )

        return {
            'fixed_code': fixed_code,
            'applied_fixes': applied_fixes,
            'is_valid': self.phase3._validate_fix(fixed_code)
        }

    def _generate_summary(self, file_results: List[Dict]) -> Dict:
        """
        Generate summary statistics
        """
        total_files = len(file_results)
        defective_files = sum(1 for r in file_results if r['phase1'] and r['phase1']['is_defective'])
        fixed_files = sum(1 for r in file_results if r.get('phase3') and r['phase3']['applied_fixes'])

        summary = {
            'total_files': total_files,
            'defective_files': defective_files,
            'clean_files': total_files - defective_files,
            'fixed_files': fixed_files,
            'defect_rate': defective_files / total_files if total_files > 0 else 0,
            'fix_rate': fixed_files / defective_files if defective_files > 0 else 0
        }

        return summary

    def save_results(self, results: Dict, output_dir: str = './framework_results'):
        """
        Save results to files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON
        json_file = os.path.join(output_dir, 'results.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {json_file}")

        # Generate report
        report = self._generate_report(results)
        report_file = os.path.join(output_dir, 'report.txt')
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to: {report_file}")

        return output_dir

    def _generate_report(self, results: Dict) -> str:
        """
        Generate text report
        """
        lines = []
        lines.append("="*80)
        lines.append("UNIFIED DEFECT MITIGATION FRAMEWORK - FINAL REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {results['timestamp']}")
        lines.append("")

        # Summary
        summary = results['summary']
        lines.append("SUMMARY")
        lines.append("-"*80)
        lines.append(f"Total files:      {summary['total_files']}")
        lines.append(f"Defective files:  {summary['defective_files']} ({summary['defect_rate']*100:.1f}%)")
        lines.append(f"Clean files:      {summary['clean_files']}")
        lines.append(f"Fixed files:      {summary['fixed_files']}")
        lines.append("")

        # File details
        lines.append("FILE DETAILS")
        lines.append("-"*80)

        for file_result in results['files']:
            lines.append(f"\nFile: {file_result['file']}")

            if file_result.get('phase1'):
                p1 = file_result['phase1']
                status = "DEFECTIVE" if p1['is_defective'] else "CLEAN"
                lines.append(f"  Phase 1: {status} ({p1['defect_probability']*100:.2f}% probability)")

            if file_result.get('phase2'):
                p2 = file_result['phase2']
                lines.append(f"  Phase 2: {len(p2['top_lines'])} suspicious lines - {p2['top_lines']}")

            if file_result.get('phase3'):
                p3 = file_result['phase3']
                lines.append(f"  Phase 3: {len(p3['applied_fixes'])} fixes applied")
                for fix in p3['applied_fixes']:
                    lines.append(f"    - {fix['type']} (confidence: {fix['confidence']:.2f})")

        lines.append("")
        lines.append("="*80)

        return "\n".join(lines)


def demo_complete_framework():
    """
    Complete demonstration of the unified framework
    """
    print("\n" + "="*80)
    print("COMPLETE FRAMEWORK DEMONSTRATION")
    print("="*80)

    # Test files
    test_files = [
        {
            'path': 'test_file_1.py',
            'code': """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Bug: no zero check

def get_first_item(items):
    return items[0]  # Bug: no empty check
"""
        },
        {
            'path': 'test_file_2.py',
            'code': """
def process_data(data):
    results = []
    for i in range(len(data)):
        value = data[i + 1]  # Bug: index out of range
        results.append(value)
    return results
"""
        },
        {
            'path': 'test_file_3.py',
            'code': """
def safe_divide(a, b):
    if b != 0:
        return a / b
    else:
        return 0  # Clean code
"""
        }
    ]

    # Initialize framework
    framework = UnifiedDefectMitigationFramework()

    # Process files
    results = framework.process(test_files)

    # Save results
    output_dir = framework.save_results(results)

    # Print summary
    print("\n" + "="*80)
    print("FRAMEWORK EXECUTION COMPLETE")
    print("="*80)
    summary = results['summary']
    print(f"Total files:      {summary['total_files']}")
    print(f"Defective files:  {summary['defective_files']}")
    print(f"Fixed files:      {summary['fixed_files']}")
    print(f"Results saved in: {output_dir}")
    print("="*80)

    return framework, results


if __name__ == "__main__":
    framework, results = demo_complete_framework()
