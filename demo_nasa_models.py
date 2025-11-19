"""
Demo script showing how to use pre-trained NASA models
"""

import numpy as np
from defect_prediction import load_nasa_model, list_available_models
from nasa_dataset_loader import NASADatasetLoader


def demo_pretrained_models():
    """Demonstrate using pre-trained NASA models"""

    print("="*70)
    print("DEMO: Using Pre-trained NASA Models")
    print("="*70)

    # List available models
    print("\nStep 1: Listing available pre-trained models...")
    models = list_available_models()
    print(f"Available models: {models}")

    if not models:
        print("\nNo models found. Please run train_nasa_models.py first.")
        return

    # Load a pre-trained model
    print("\nStep 2: Loading pre-trained JM1 model...")
    model = load_nasa_model('JM1')

    # Load test data
    print("\nStep 3: Loading JM1 test data...")
    loader = NASADatasetLoader()
    X, y, info = loader.load_dataset('JM1')

    # Take a sample for testing
    sample_indices = np.random.choice(X.shape[0], size=100, replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    print(f"  Test sample size: {X_sample.shape[0]}")
    print(f"  Actual defective: {np.sum(y_sample)} ({np.mean(y_sample)*100:.1f}%)")

    # Make predictions
    print("\nStep 4: Making predictions...")
    predictions, probabilities = model.predict(X_sample)

    print(f"  Predicted defective: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)")

    # Evaluate
    print("\nStep 5: Evaluating model...")
    results = model.evaluate(X_sample, y_sample)

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")

    # Show some example predictions
    print("\nStep 6: Example predictions (first 10 samples)...")
    print("-"*70)
    print(f"{'Index':<8} {'Actual':<10} {'Predicted':<12} {'Defect Prob':<15}")
    print("-"*70)

    for i in range(min(10, len(predictions))):
        actual = "Defective" if y_sample[i] == 1 else "Clean"
        predicted = "Defective" if predictions[i] == 1 else "Clean"
        prob = probabilities[i][1] if len(probabilities[i]) > 1 else probabilities[i][0]

        print(f"{i:<8} {actual:<10} {predicted:<12} {prob:.4f}")

    print("-"*70)


def demo_all_models():
    """Test all available models"""

    print("\n" + "="*70)
    print("DEMO: Testing All Pre-trained Models")
    print("="*70)

    # Get available models
    models = list_available_models()

    if not models:
        print("\nNo models found. Please run train_nasa_models.py first.")
        return

    loader = NASADatasetLoader()
    results_summary = []

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Testing {model_name} model")
        print('='*70)

        try:
            # Load model
            model = load_nasa_model(model_name)

            # Load corresponding dataset
            X, y, info = loader.load_dataset(model_name)

            # Take a random sample (20% of data)
            sample_size = max(100, int(X.shape[0] * 0.2))
            sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Evaluate
            results = model.evaluate(X_sample, y_sample)

            print(f"\nResults on {sample_size} samples:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1_score']:.4f}")

            results_summary.append({
                'model': model_name,
                'samples': sample_size,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            })

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL MODELS")
    print("="*70)
    print(f"{'Model':<10} {'Samples':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)

    for r in results_summary:
        print(f"{r['model']:<10} {r['samples']:<10} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1_score']:<12.4f}")

    print("="*70)


if __name__ == "__main__":
    # Run demo
    demo_pretrained_models()

    # Test all models
    print("\n\n")
    demo_all_models()
