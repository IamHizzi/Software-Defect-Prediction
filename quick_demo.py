"""
Quick Demo - Analyze Code for Defects
Shows different ways to use the defect prediction system
"""

import sys
from analyze_code import CodeDefectAnalyzer

def demo_single_file():
    """Demo: Analyze a single file"""
    print("\n" + "="*70)
    print("DEMO 1: Analyzing a Single File")
    print("="*70)

    analyzer = CodeDefectAnalyzer(model_name='JM1', use_unified_framework=False)

    # Analyze the buggy processor file
    result = analyzer.analyze_file('sample_code/buggy_processor.py')

    if result:
        print(f"\n✓ Analysis complete!")
        print(f"  File is: {result['prediction']['status']}")
        print(f"  Defect probability: {result['prediction']['defect_probability']:.2%}")


def demo_directory():
    """Demo: Analyze all files in a directory"""
    print("\n" + "="*70)
    print("DEMO 2: Analyzing a Directory")
    print("="*70)

    analyzer = CodeDefectAnalyzer(model_name='KC1', use_unified_framework=False)

    # Analyze all Python files
    results = analyzer.analyze_directory('sample_code')

    print(f"\n✓ Analysis complete!")
    print(f"  Total files: {len(results)}")
    defective = sum(1 for r in results if r['prediction']['is_defective'])
    print(f"  Defective: {defective}")
    print(f"  Clean: {len(results) - defective}")

    # Save results
    analyzer.save_results(results, output_dir='./demo_results')


def demo_inline_code():
    """Demo: Analyze code directly (not from file)"""
    print("\n" + "="*70)
    print("DEMO 3: Analyzing Code Snippet Directly")
    print("="*70)

    # Some code with potential defects
    code = """
def risky_function(data):
    result = []
    for i in range(len(data)):
        value = data[i + 1]  # Bug: index out of range
        result.append(value)
    return result

def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Bug: no zero check
"""

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    analyzer = CodeDefectAnalyzer(model_name='JM1', use_unified_framework=False)
    result = analyzer.analyze_file(temp_file)

    # Cleanup
    import os
    os.unlink(temp_file)

    if result:
        print(f"\n✓ Analysis complete!")
        print(f"  Status: {result['prediction']['status']}")
        print(f"  Defect probability: {result['prediction']['defect_probability']:.2%}")
        print(f"  LOC: {result['metrics']['loc']}")
        print(f"  Complexity: {result['metrics']['cyclomatic_complexity']}")


def demo_compare_models():
    """Demo: Compare different NASA models"""
    print("\n" + "="*70)
    print("DEMO 4: Comparing Different Models")
    print("="*70)

    test_file = 'sample_code/file_handler.py'

    for model_name in ['CM1', 'JM1', 'KC1', 'KC2', 'PC1']:
        try:
            analyzer = CodeDefectAnalyzer(model_name=model_name, use_unified_framework=False)

            # Quick analysis without full output
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Read file
                with open(test_file, 'r') as f:
                    code = f.read()

                # Extract metrics
                from defect_prediction import extract_software_metrics
                import numpy as np

                metrics = extract_software_metrics(code)
                metrics_vector = np.array(list(metrics.values())).reshape(1, -1)

                # Pad to 21 features
                if metrics_vector.shape[1] < 21:
                    padding = np.zeros((1, 21 - metrics_vector.shape[1]))
                    metrics_vector = np.hstack([metrics_vector, padding])

                # Predict
                predictions, probabilities = analyzer.predictor.predict(metrics_vector)
                defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]

                print(f"  {model_name}: {defect_prob:.2%} defect probability")

        except Exception as e:
            print(f"  {model_name}: Error - {e}")


def demo_generate_report():
    """Demo: Generate comprehensive report"""
    print("\n" + "="*70)
    print("DEMO 5: Generating Comprehensive Report")
    print("="*70)

    analyzer = CodeDefectAnalyzer(model_name='JM1', use_unified_framework=False)

    # Analyze directory
    results = analyzer.analyze_directory('sample_code')

    # Generate report
    report = analyzer.generate_report(results)

    print("\n--- GENERATED REPORT ---")
    print(report)


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("NASA DEFECT PREDICTION - QUICK DEMO")
    print("="*70)
    print("\nThis demo shows how to use the defect prediction system")

    try:
        # Run demos
        demo_single_file()
        input("\nPress Enter to continue to next demo...")

        demo_directory()
        input("\nPress Enter to continue to next demo...")

        demo_inline_code()
        input("\nPress Enter to continue to next demo...")

        demo_compare_models()
        input("\nPress Enter to continue to next demo...")

        demo_generate_report()

        print("\n" + "="*70)
        print("✓ ALL DEMOS COMPLETE!")
        print("="*70)
        print("\nYou can now use the analyzer in your own projects:")
        print("  python analyze_code.py <file_or_directory> --model JM1")
        print("\nFor more options:")
        print("  python analyze_code.py --help")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError in demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
