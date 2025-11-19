"""
Training Script for NASA Promise Datasets
Trains defect prediction models on NASA datasets and saves results
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modules
from nasa_dataset_loader import NASADatasetLoader, get_dataset_summary
from defect_prediction import DefectPredictor


class NASAModelTrainer:
    """
    Trainer for NASA dataset models
    """

    def __init__(self, output_dir='./models'):
        """
        Initialize trainer

        Args:
            output_dir: Directory to save trained models and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        self.models_dir = os.path.join(output_dir, 'trained_models')
        self.results_dir = os.path.join(output_dir, 'results')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def train_single_dataset(self, dataset_name, X, y, n_features=20, test_size=0.2):
        """
        Train model on a single dataset

        Args:
            dataset_name: Name of the dataset
            X: Features
            y: Labels
            n_features: Number of features to select
            test_size: Test set size

        Returns:
            Dictionary with model and results
        """
        print(f"\n{'='*70}")
        print(f"Training Model on {dataset_name}")
        print('='*70)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set:  {X_test.shape[0]} samples")
        print(f"Train defect rate: {np.mean(y_train)*100:.1f}%")
        print(f"Test defect rate:  {np.mean(y_test)*100:.1f}%")

        # Determine optimal number of features
        n_features = min(n_features, X.shape[1])

        # Train model
        print(f"\nTraining ensemble model with {n_features} features...")
        predictor = DefectPredictor(n_features=n_features)
        predictor.train(X_train, y_train)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = predictor.evaluate(X_test, y_test)

        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")

        # Cross-validation on full dataset
        print("\nPerforming 5-fold cross-validation...")
        try:
            # Scale and select features for CV
            X_scaled = predictor.scaler.transform(X)
            X_selected = predictor.feature_selector.transform(X_scaled)

            cv_scores = cross_val_score(
                predictor.ensemble, X_selected, y,
                cv=5, scoring='f1', n_jobs=-1
            )

            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            results['cv_scores'] = cv_scores.tolist()
            results['cv_mean'] = float(cv_scores.mean())
            results['cv_std'] = float(cv_scores.std())
        except Exception as e:
            print(f"  Warning: Cross-validation failed: {e}")
            results['cv_scores'] = None
            results['cv_mean'] = None
            results['cv_std'] = None

        # Save model
        model_path = os.path.join(self.models_dir, f'{dataset_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        print(f"\nModel saved to: {model_path}")

        # Prepare results summary
        results_summary = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_selected_features': n_features,
            'test_size': test_size,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'metrics': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
            },
            'cross_validation': {
                'scores': results['cv_scores'],
                'mean': results['cv_mean'],
                'std': results['cv_std']
            }
        }

        return {
            'model': predictor,
            'results': results_summary,
            'X_test': X_test,
            'y_test': y_test
        }

    def train_all_datasets(self, datasets, n_features=20):
        """
        Train models on all datasets

        Args:
            datasets: Dictionary of datasets from NASADatasetLoader
            n_features: Number of features to select

        Returns:
            Dictionary with all training results
        """
        all_results = {}

        print(f"\n{'='*70}")
        print(f"TRAINING MODELS ON {len(datasets)} NASA DATASETS")
        print('='*70)

        for dataset_name, (X, y, info) in datasets.items():
            try:
                result = self.train_single_dataset(
                    dataset_name, X, y, n_features=n_features
                )
                all_results[dataset_name] = result

            except Exception as e:
                print(f"\nError training on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_results

    def save_results(self, all_results, datasets):
        """
        Save training results to files

        Args:
            all_results: Dictionary of training results
            datasets: Original datasets dictionary
        """
        # Save detailed results as JSON
        results_json = {}
        for dataset_name, result in all_results.items():
            results_json[dataset_name] = result['results']

        json_path = os.path.join(self.results_dir, 'training_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")

        # Create summary table
        summary_data = []
        for dataset_name, result in all_results.items():
            metrics = result['results']['metrics']
            cv = result['results']['cross_validation']

            summary_data.append({
                'Dataset': dataset_name,
                'Samples': result['results']['n_samples'],
                'Features': result['results']['n_selected_features'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV F1 Mean': f"{cv['mean']:.4f}" if cv['mean'] else 'N/A',
                'CV F1 Std': f"{cv['std']:.4f}" if cv['std'] else 'N/A'
            })

        summary_df = pd.DataFrame(summary_data)

        # Save as CSV
        csv_path = os.path.join(self.results_dir, 'training_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")

        return summary_df

    def generate_report(self, all_results, datasets):
        """
        Generate comprehensive training report

        Args:
            all_results: Dictionary of training results
            datasets: Original datasets dictionary

        Returns:
            Report text
        """
        report = []
        report.append("="*70)
        report.append("NASA PROMISE DATASET - MODEL TRAINING REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset summary
        report.append("DATASET SUMMARY")
        report.append("-"*70)
        summary = get_dataset_summary(datasets)
        report.append(summary.to_string(index=False))
        report.append("")

        # Training results
        report.append("TRAINING RESULTS")
        report.append("-"*70)

        for dataset_name, result in all_results.items():
            report.append(f"\n{dataset_name}:")
            metrics = result['results']['metrics']
            cv = result['results']['cross_validation']

            report.append(f"  Test Set Metrics:")
            report.append(f"    Accuracy:  {metrics['accuracy']:.4f}")
            report.append(f"    Precision: {metrics['precision']:.4f}")
            report.append(f"    Recall:    {metrics['recall']:.4f}")
            report.append(f"    F1-Score:  {metrics['f1_score']:.4f}")

            if cv['mean']:
                report.append(f"  Cross-Validation:")
                report.append(f"    Mean F1:   {cv['mean']:.4f}")
                report.append(f"    Std F1:    {cv['std']:.4f}")

        report.append("")
        report.append("="*70)

        # Overall statistics
        report.append("\nOVERALL STATISTICS")
        report.append("-"*70)

        avg_f1 = np.mean([r['results']['metrics']['f1_score'] for r in all_results.values()])
        avg_precision = np.mean([r['results']['metrics']['precision'] for r in all_results.values()])
        avg_recall = np.mean([r['results']['metrics']['recall'] for r in all_results.values()])
        avg_accuracy = np.mean([r['results']['metrics']['accuracy'] for r in all_results.values()])

        report.append(f"Average Accuracy:  {avg_accuracy:.4f}")
        report.append(f"Average Precision: {avg_precision:.4f}")
        report.append(f"Average Recall:    {avg_recall:.4f}")
        report.append(f"Average F1-Score:  {avg_f1:.4f}")

        report.append("\n" + "="*70)
        report.append(f"Models saved in: {self.models_dir}")
        report.append(f"Results saved in: {self.results_dir}")
        report.append("="*70)

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(self.results_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\nReport saved to: {report_path}")

        return report_text


def main():
    """Main training pipeline"""

    print("\n" + "="*70)
    print("NASA PROMISE DATASET - MODEL TRAINING")
    print("="*70)

    # Step 1: Load datasets
    print("\nStep 1: Loading NASA datasets...")
    loader = NASADatasetLoader(cache_dir='./nasa_datasets')
    datasets = loader.load_all_datasets(['CM1', 'JM1', 'KC1', 'KC2', 'PC1'])

    if not datasets:
        print("Error: No datasets loaded. Exiting.")
        return

    # Step 2: Train models
    print("\nStep 2: Training models...")
    trainer = NASAModelTrainer(output_dir='./models')
    all_results = trainer.train_all_datasets(datasets, n_features=20)

    if not all_results:
        print("Error: No models trained. Exiting.")
        return

    # Step 3: Save results
    print("\nStep 3: Saving results...")
    summary_df = trainer.save_results(all_results, datasets)

    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    report = trainer.generate_report(all_results, datasets)

    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Trained {len(all_results)} models")
    print(f"Models directory: {trainer.models_dir}")
    print(f"Results directory: {trainer.results_dir}")


if __name__ == "__main__":
    main()
