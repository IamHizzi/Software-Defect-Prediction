"""
Complete End-to-End Defect Prediction Pipeline Demonstration

This script demonstrates the complete flow:
1. Dataset Acquisition & Preparation
2. Model Training
3. Phase 1: Defect Prediction
4. Phase 2: Defect Localization
5. Phase 3: Bug Fix Generation

Professional implementation with comprehensive results
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CompletePipelineDemo:
    """
    Complete end-to-end pipeline demonstration
    """

    def __init__(self):
        self.output_dir = './complete_pipeline_results'
        os.makedirs(self.output_dir, exist_ok=True)

        self.phase1_results = None
        self.phase2_results = None
        self.phase3_results = None

    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print('='*80)

    def print_step(self, step_num, step_name):
        """Print step header"""
        print(f"\n{'─'*80}")
        print(f"  STEP {step_num}: {step_name}")
        print('─'*80)

    def phase_0_dataset_acquisition(self):
        """PHASE 0: Dataset Acquisition & Preparation"""

        self.print_header("PHASE 0: DATASET ACQUISITION & PREPARATION")

        from nasa_dataset_loader import NASADatasetLoader, get_dataset_summary

        self.print_step(1, "Initialize Dataset Loader")
        loader = NASADatasetLoader(cache_dir='./nasa_datasets')
        print("✓ Dataset loader initialized")

        self.print_step(2, "Download NASA Promise Datasets")
        print("  Downloading datasets: CM1, JM1, KC1, KC2, PC1")
        datasets = loader.load_all_datasets(['CM1', 'JM1', 'KC1', 'KC2', 'PC1'])

        if not datasets:
            print("❌ Failed to load datasets")
            return None

        print(f"✓ Successfully loaded {len(datasets)} datasets")

        self.print_step(3, "Dataset Summary")
        summary = get_dataset_summary(datasets)
        print(summary.to_string(index=False))

        # Save dataset info
        dataset_info = {
            'total_datasets': len(datasets),
            'total_samples': sum(info['n_samples'] for _, _, info in datasets.values()),
            'datasets': {name: info for name, (_, _, info) in datasets.items()}
        }

        with open(os.path.join(self.output_dir, 'phase0_datasets.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\n✓ PHASE 0 COMPLETE")
        print(f"  Total datasets: {len(datasets)}")
        print(f"  Total samples:  {dataset_info['total_samples']:,}")

        return datasets

    def phase_0b_model_training(self, datasets):
        """PHASE 0B: Model Training"""

        self.print_header("PHASE 0B: MODEL TRAINING")

        # Check if enhanced training exists
        try:
            from enhanced_training import EnhancedTrainingPipeline

            self.print_step(1, "Initialize Enhanced Training Pipeline")
            pipeline = EnhancedTrainingPipeline(output_dir='./enhanced_models')
            print("✓ Training pipeline initialized")

            self.print_step(2, "Train Enhanced Models")
            print("  Training advanced stacking ensemble on each dataset...")
            print("  This may take several minutes...")

            results = pipeline.train_all_datasets(datasets)

            self.print_step(3, "Save Training Results")
            summary = pipeline.save_results(results)
            print(summary.to_string(index=False))

            print(f"\n✓ PHASE 0B COMPLETE")
            print(f"  Models trained: {len(results)}")
            print(f"  Models saved in: {pipeline.models_dir}")

            return results

        except ImportError:
            print("⚠ Enhanced training not available, using standard training")

            from train_nasa_models import NASAModelTrainer

            trainer = NASAModelTrainer(output_dir='./models')
            results = trainer.train_all_datasets(datasets, n_features=15)
            summary = trainer.save_results(results, datasets)

            print(f"\n✓ PHASE 0B COMPLETE (Standard Training)")
            return results

    def phase_1_defect_prediction(self, test_files):
        """PHASE 1: Defect Prediction"""

        self.print_header("PHASE 1: DEFECT PREDICTION")

        from defect_prediction import extract_software_metrics
        import pickle

        self.print_step(1, "Load Trained Model")

        # Try to load enhanced model first, fallback to standard
        model_paths = [
            './enhanced_models/models/JM1_enhanced.pkl',
            './models/trained_models/JM1_model.pkl'
        ]

        model = None
        model_type = None
        for path in model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                model_type = 'Enhanced' if 'enhanced' in path else 'Standard'
                print(f"✓ Loaded {model_type} JM1 model from: {path}")
                break

        if model is None:
            print("❌ No trained model found")
            return None

        self.print_step(2, "Analyze Test Files")

        results = []
        for file_path in test_files:
            print(f"\n  Analyzing: {file_path}")

            # Read code
            with open(file_path, 'r') as f:
                code = f.read()

            # Extract metrics
            metrics = extract_software_metrics(code)

            # Prepare feature vector
            metrics_vector = np.array(list(metrics.values())).reshape(1, -1)
            if metrics_vector.shape[1] < 21:
                padding = np.zeros((1, 21 - metrics_vector.shape[1]))
                metrics_vector = np.hstack([metrics_vector, padding])
            elif metrics_vector.shape[1] > 21:
                metrics_vector = metrics_vector[:, :21]

            # Predict
            predictions, probabilities = model.predict(metrics_vector)
            is_defective = predictions[0] == 1
            defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]

            # Store result
            result = {
                'file': os.path.basename(file_path),
                'path': file_path,
                'is_defective': bool(is_defective),
                'defect_probability': float(defect_prob),
                'metrics': metrics
            }
            results.append(result)

            # Display
            status = "⚠️  DEFECTIVE" if is_defective else "✓  CLEAN"
            print(f"    Status: {status}")
            print(f"    Probability: {defect_prob*100:.2f}%")
            print(f"    LOC: {metrics.get('loc', 0)}")
            print(f"    Complexity: {metrics.get('cyclomatic_complexity', 0)}")

        # Save results
        phase1_file = os.path.join(self.output_dir, 'phase1_predictions.json')
        with open(phase1_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ PHASE 1 COMPLETE")
        print(f"  Files analyzed: {len(results)}")
        print(f"  Defective: {sum(1 for r in results if r['is_defective'])}")
        print(f"  Clean: {sum(1 for r in results if not r['is_defective'])}")

        self.phase1_results = results
        return results

    def phase_2_defect_localization(self, phase1_results):
        """PHASE 2: Defect Localization"""

        self.print_header("PHASE 2: DEFECT LOCALIZATION")

        try:
            from defect_localization import DefectLocalizer
        except ImportError as e:
            print(f"⚠ Defect localization not available: {e}")
            print("  Skipping Phase 2")
            return None

        self.print_step(1, "Initialize Localizer")
        localizer = DefectLocalizer(hidden_channels=64, num_heads=4)
        print("✓ Localizer initialized")

        self.print_step(2, "Localize Defects in Flagged Files")

        results = []
        for prediction in phase1_results:
            if not prediction['is_defective']:
                continue

            print(f"\n  Localizing: {prediction['file']}")

            # Read code
            with open(prediction['path'], 'r') as f:
                code = f.read()

            # Localize
            try:
                localization = localizer.localize_defects(
                    code,
                    defect_prob=prediction['defect_probability'],
                    metrics=prediction['metrics']
                )

                defective_lines = localizer.get_defective_lines(
                    code,
                    defect_prob=prediction['defect_probability'],
                    metrics=prediction['metrics']
                )

                result = {
                    'file': prediction['file'],
                    'defective_lines': defective_lines,
                    'num_suspicious_nodes': len(localization['defective_nodes']),
                    'top_nodes': localization['top_nodes'][:5]
                }
                results.append(result)

                print(f"    Defective lines: {defective_lines}")
                print(f"    Suspicious nodes: {len(localization['defective_nodes'])}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Save results
        phase2_file = os.path.join(self.output_dir, 'phase2_localization.json')
        with open(phase2_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ PHASE 2 COMPLETE")
        print(f"  Files localized: {len(results)}")

        self.phase2_results = results
        return results

    def phase_3_bug_fix_generation(self, phase1_results, phase2_results):
        """PHASE 3: Bug Fix Generation"""

        self.print_header("PHASE 3: BUG FIX GENERATION")

        try:
            from bug_fix import BugFixer
        except ImportError as e:
            print(f"⚠ Bug fixer not available: {e}")
            print("  Skipping Phase 3")
            return None

        self.print_step(1, "Initialize Bug Fixer")
        fixer = BugFixer()
        print("✓ Bug fixer initialized")

        self.print_step(2, "Generate Fixes for Defective Files")

        results = []

        # Match phase1 and phase2 results
        phase2_dict = {r['file']: r for r in (phase2_results or [])}

        for prediction in phase1_results:
            if not prediction['is_defective']:
                continue

            print(f"\n  Generating fix: {prediction['file']}")

            # Read code
            with open(prediction['path'], 'r') as f:
                code = f.read()

            # Get defective lines
            defective_lines = []
            if prediction['file'] in phase2_dict:
                defective_lines = phase2_dict[prediction['file']]['defective_lines']

            # Generate fix
            try:
                fixed_code, applied_fixes = fixer.generate_fix(
                    code,
                    buggy_lines=defective_lines
                )

                result = {
                    'file': prediction['file'],
                    'original_path': prediction['path'],
                    'fixes_applied': len(applied_fixes),
                    'fix_details': applied_fixes,
                    'fixed_code': fixed_code
                }
                results.append(result)

                # Save fixed code
                fixed_path = os.path.join(
                    self.output_dir,
                    f"fixed_{prediction['file']}"
                )
                with open(fixed_path, 'w') as f:
                    f.write(fixed_code)

                print(f"    Fixes applied: {len(applied_fixes)}")
                for fix in applied_fixes:
                    print(f"      - {fix.get('type', 'unknown')}")
                print(f"    Fixed code saved: {fixed_path}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Save results
        phase3_file = os.path.join(self.output_dir, 'phase3_fixes.json')

        # Create JSON-serializable version
        json_results = []
        for r in results:
            json_r = {
                'file': r['file'],
                'original_path': r['original_path'],
                'fixes_applied': r['fixes_applied'],
                'fix_details': [
                    {'type': f.get('type'), 'confidence': f.get('similarity', 0)}
                    for f in r['fix_details']
                ]
            }
            json_results.append(json_r)

        with open(phase3_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ PHASE 3 COMPLETE")
        print(f"  Files fixed: {len(results)}")

        self.phase3_results = results
        return results

    def generate_final_report(self):
        """Generate comprehensive final report"""

        self.print_header("FINAL REPORT GENERATION")

        report = []
        report.append("="*80)
        report.append("COMPLETE DEFECT PREDICTION PIPELINE - FINAL REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Phase 1 Summary
        if self.phase1_results:
            report.append("PHASE 1: DEFECT PREDICTION")
            report.append("-"*80)
            defective = sum(1 for r in self.phase1_results if r['is_defective'])
            report.append(f"  Total files analyzed:  {len(self.phase1_results)}")
            report.append(f"  Defective files:       {defective}")
            report.append(f"  Clean files:           {len(self.phase1_results) - defective}")
            report.append("")

            for result in self.phase1_results:
                report.append(f"  {result['file']}:")
                report.append(f"    Status: {'DEFECTIVE' if result['is_defective'] else 'CLEAN'}")
                report.append(f"    Probability: {result['defect_probability']*100:.2f}%")
                report.append("")

        # Phase 2 Summary
        if self.phase2_results:
            report.append("PHASE 2: DEFECT LOCALIZATION")
            report.append("-"*80)
            report.append(f"  Files localized: {len(self.phase2_results)}")
            report.append("")

            for result in self.phase2_results:
                report.append(f"  {result['file']}:")
                report.append(f"    Defective lines: {result['defective_lines']}")
                report.append(f"    Suspicious nodes: {result['num_suspicious_nodes']}")
                report.append("")

        # Phase 3 Summary
        if self.phase3_results:
            report.append("PHASE 3: BUG FIX GENERATION")
            report.append("-"*80)
            report.append(f"  Files fixed: {len(self.phase3_results)}")
            report.append("")

            for result in self.phase3_results:
                report.append(f"  {result['file']}:")
                report.append(f"    Fixes applied: {result['fixes_applied']}")
                if result['fix_details']:
                    report.append(f"    Fix types:")
                    for fix in result['fix_details']:
                        report.append(f"      - {fix.get('type', 'unknown')}")
                report.append("")

        report.append("="*80)

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(self.output_dir, 'FINAL_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Report saved: {report_path}")

        return report_text

    def run_complete_pipeline(self, test_files):
        """Run the complete pipeline"""

        self.print_header("COMPLETE END-TO-END PIPELINE DEMONSTRATION")

        print("\nThis demonstration will show the complete flow:")
        print("  0. Dataset Acquisition & Preparation")
        print("  0B. Model Training")
        print("  1. Defect Prediction")
        print("  2. Defect Localization")
        print("  3. Bug Fix Generation")
        print("\nPress Ctrl+C to cancel...")

        try:
            # Phase 0: Dataset Acquisition
            datasets = self.phase_0_dataset_acquisition()
            if not datasets:
                return

            input("\nPress Enter to continue to Model Training...")

            # Phase 0B: Model Training
            training_results = self.phase_0b_model_training(datasets)
            if not training_results:
                return

            input("\nPress Enter to continue to Defect Prediction...")

            # Phase 1: Defect Prediction
            phase1_results = self.phase_1_defect_prediction(test_files)
            if not phase1_results:
                return

            input("\nPress Enter to continue to Defect Localization...")

            # Phase 2: Defect Localization
            phase2_results = self.phase_2_defect_localization(phase1_results)

            input("\nPress Enter to continue to Bug Fix Generation...")

            # Phase 3: Bug Fix Generation
            phase3_results = self.phase_3_bug_fix_generation(phase1_results, phase2_results)

            input("\nPress Enter to generate final report...")

            # Final Report
            self.generate_final_report()

            print(f"\n{'='*80}")
            print("✓ COMPLETE PIPELINE DEMONSTRATION FINISHED")
            print('='*80)
            print(f"All results saved in: {self.output_dir}")

        except KeyboardInterrupt:
            print("\n\n⚠ Pipeline interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""

    # Get test files
    test_files = [
        'sample_code/buggy_processor.py',
        'sample_code/file_handler.py',
        'sample_code/string_utils.py',
        'sample_code/calculator.py'
    ]

    # Verify files exist
    test_files = [f for f in test_files if os.path.exists(f)]

    if not test_files:
        print("❌ No test files found in sample_code/")
        return 1

    # Run complete pipeline
    demo = CompletePipelineDemo()
    demo.run_complete_pipeline(test_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
