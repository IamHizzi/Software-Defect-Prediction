"""
Code Defect Analysis Script
Analyzes Python code files and predicts defects using NASA-trained models
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from defect_prediction import load_nasa_model, extract_software_metrics, list_available_models


class CodeDefectAnalyzer:
    """
    Analyzes code files for defects using trained NASA models
    """

    def __init__(self, model_name='JM1', use_unified_framework=True):
        """
        Initialize the analyzer

        Args:
            model_name: Which NASA model to use (CM1, JM1, KC1, KC2, PC1)
            use_unified_framework: Whether to use full framework (prediction + localization + fix)
        """
        self.model_name = model_name
        self.use_unified_framework = use_unified_framework

        print(f"Initializing Code Defect Analyzer...")
        print(f"  Model: {model_name}")
        print(f"  Framework: {'Unified (Full Pipeline)' if use_unified_framework else 'Prediction Only'}")

        # Load the prediction model
        try:
            self.predictor = load_nasa_model(model_name)
            print(f"  ✓ Loaded {model_name} model")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            print(f"  Available models: {list_available_models()}")
            raise

        # Initialize unified framework if requested
        if use_unified_framework:
            try:
                from unified_framework import UnifiedDefectMitigationFramework
                self.framework = UnifiedDefectMitigationFramework()
                print(f"  ✓ Initialized unified framework")
            except ImportError as e:
                print(f"  ⚠ Unified framework not available: {e}")
                print(f"  ℹ Running in prediction-only mode")
                self.framework = None
                self.use_unified_framework = False
        else:
            self.framework = None

    def analyze_file(self, file_path):
        """
        Analyze a single Python file

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {file_path}")
        print('='*70)

        # Read file
        try:
            with open(file_path, 'r') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

        print(f"  Lines of code: {len(code.split(chr(10)))}")

        # Extract metrics
        metrics = extract_software_metrics(code)
        print(f"\n  Software Metrics:")
        for key, value in metrics.items():
            print(f"    {key:25s}: {value}")

        # Prepare feature vector (pad to 21 features to match NASA dataset structure)
        metrics_vector = np.array(list(metrics.values())).reshape(1, -1)
        if metrics_vector.shape[1] < 21:
            padding = np.zeros((1, 21 - metrics_vector.shape[1]))
            metrics_vector = np.hstack([metrics_vector, padding])
        elif metrics_vector.shape[1] > 21:
            # Trim to 21 features if we have more
            metrics_vector = metrics_vector[:, :21]

        # Predict using the model
        predictions, probabilities = self.predictor.predict(metrics_vector)

        is_defective = predictions[0] == 1
        defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]

        print(f"\n  Prediction Results:")
        print(f"    Status: {'⚠️  DEFECTIVE' if is_defective else '✓  CLEAN'}")
        print(f"    Defect Probability: {defect_prob:.2%}")
        print(f"    Model: {self.model_name}")

        # Build result dictionary
        result = {
            'file_path': file_path,
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'metrics': metrics,
            'prediction': {
                'is_defective': bool(is_defective),
                'defect_probability': float(defect_prob),
                'status': 'DEFECTIVE' if is_defective else 'CLEAN'
            },
            'code': code
        }

        # Use unified framework if enabled and defect detected
        if self.use_unified_framework and is_defective:
            print(f"\n  Running Full Defect Analysis Pipeline...")
            framework_result = self.framework.process_code_file(code, file_path)
            result['localization'] = framework_result.get('phase2_localization')
            result['fix'] = framework_result.get('phase3_fix')
            result['fixed_code'] = framework_result.get('fixed_code')
        else:
            result['localization'] = None
            result['fix'] = None
            result['fixed_code'] = None

        return result

    def analyze_directory(self, directory_path, pattern="*.py"):
        """
        Analyze all Python files in a directory

        Args:
            directory_path: Path to directory
            pattern: File pattern to match (default: *.py)

        Returns:
            List of results for all files
        """
        print(f"\n{'='*70}")
        print(f"Analyzing Directory: {directory_path}")
        print(f"  Pattern: {pattern}")
        print('='*70)

        # Find all matching files
        path = Path(directory_path)
        files = list(path.glob(f"**/{pattern}"))

        print(f"  Found {len(files)} files to analyze")

        results = []
        for file_path in files:
            result = self.analyze_file(str(file_path))
            if result:
                results.append(result)

        return results

    def generate_report(self, results, output_file=None):
        """
        Generate analysis report

        Args:
            results: List of analysis results
            output_file: Optional path to save report

        Returns:
            Report text
        """
        if not results:
            return "No results to report"

        report = []
        report.append("="*70)
        report.append("CODE DEFECT ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Used: {self.model_name}")
        report.append(f"Files Analyzed: {len(results)}")
        report.append("")

        # Summary statistics
        defective_count = sum(1 for r in results if r['prediction']['is_defective'])
        clean_count = len(results) - defective_count
        avg_prob = np.mean([r['prediction']['defect_probability'] for r in results])

        report.append("SUMMARY")
        report.append("-"*70)
        report.append(f"  Total Files:      {len(results)}")
        report.append(f"  Defective:        {defective_count} ({defective_count/len(results)*100:.1f}%)")
        report.append(f"  Clean:            {clean_count} ({clean_count/len(results)*100:.1f}%)")
        report.append(f"  Avg Defect Prob:  {avg_prob:.2%}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-"*70)

        for i, result in enumerate(results, 1):
            report.append(f"\n{i}. {result['file_path']}")
            report.append(f"   Status: {result['prediction']['status']}")
            report.append(f"   Defect Probability: {result['prediction']['defect_probability']:.2%}")

            # Key metrics
            metrics = result['metrics']
            report.append(f"   Key Metrics:")
            report.append(f"     LOC: {metrics.get('loc', 0)}")
            report.append(f"     Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 0)}")
            report.append(f"     Max Nesting Depth: {metrics.get('max_nesting_depth', 0)}")
            report.append(f"     Functions: {metrics.get('num_functions', 0)}")

            # Localization info if available
            if result.get('localization'):
                loc = result['localization']
                if loc['defective_lines']:
                    report.append(f"   Defective Lines: {loc['defective_lines']}")

            # Fix info if available
            if result.get('fix'):
                fix = result['fix']
                if fix['applied_fixes']:
                    report.append(f"   Fixes Applied: {fix['num_fixes_applied']}")
                    for f in fix['applied_fixes']:
                        report.append(f"     - {f.get('type', 'unknown')}")

        report.append("\n" + "="*70)

        report_text = "\n".join(report)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")

        return report_text

    def save_results(self, results, output_dir="./analysis_results"):
        """
        Save results to files

        Args:
            results: List of analysis results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON
        json_file = os.path.join(output_dir, f'analysis_{timestamp}.json')
        json_data = []
        for r in results:
            # Create JSON-serializable version
            json_r = {
                'file_path': r['file_path'],
                'timestamp': r['timestamp'],
                'model': r['model'],
                'metrics': r['metrics'],
                'prediction': r['prediction'],
            }
            if r.get('localization'):
                json_r['localization'] = r['localization']
            if r.get('fix') and r['fix'].get('applied_fixes'):
                json_r['fixes'] = [
                    {'type': f.get('type'), 'confidence': f.get('similarity')}
                    for f in r['fix']['applied_fixes']
                ]
            json_data.append(json_r)

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_file}")

        # Save report
        report_file = os.path.join(output_dir, f'report_{timestamp}.txt')
        report = self.generate_report(results, output_file=report_file)

        # Save CSV summary
        csv_file = os.path.join(output_dir, f'summary_{timestamp}.csv')
        summary_data = []
        for r in results:
            summary_data.append({
                'File': os.path.basename(r['file_path']),
                'Status': r['prediction']['status'],
                'Defect Probability': f"{r['prediction']['defect_probability']:.2%}",
                'LOC': r['metrics'].get('loc', 0),
                'Complexity': r['metrics'].get('cyclomatic_complexity', 0),
                'Functions': r['metrics'].get('num_functions', 0),
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        print(f"CSV summary saved to: {csv_file}")

        # Save fixed code if available
        for r in results:
            if r.get('fixed_code') and r['fixed_code'] != r['code']:
                base_name = os.path.basename(r['file_path'])
                fixed_file = os.path.join(output_dir, f'fixed_{base_name}')
                with open(fixed_file, 'w') as f:
                    f.write(r['fixed_code'])
                print(f"Fixed code saved to: {fixed_file}")

        return output_dir


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Python code for defects')
    parser.add_argument('path', help='File or directory to analyze')
    parser.add_argument('--model', default='JM1', choices=['CM1', 'JM1', 'KC1', 'KC2', 'PC1'],
                       help='NASA model to use (default: JM1)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple prediction only (no localization/fixing)')
    parser.add_argument('--output', default='./analysis_results',
                       help='Output directory for results (default: ./analysis_results)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CodeDefectAnalyzer(
        model_name=args.model,
        use_unified_framework=not args.simple
    )

    # Analyze
    if os.path.isfile(args.path):
        # Single file
        results = [analyzer.analyze_file(args.path)]
    elif os.path.isdir(args.path):
        # Directory
        results = analyzer.analyze_directory(args.path)
    else:
        print(f"Error: Path not found: {args.path}")
        return 1

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("\nNo files were successfully analyzed.")
        return 1

    # Save results
    print(f"\n{'='*70}")
    print("Saving Results...")
    print('='*70)
    analyzer.save_results(results, output_dir=args.output)

    # Print summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print('='*70)
    print(f"Files analyzed: {len(results)}")
    defective = sum(1 for r in results if r['prediction']['is_defective'])
    print(f"Defective files: {defective}")
    print(f"Clean files: {len(results) - defective}")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
