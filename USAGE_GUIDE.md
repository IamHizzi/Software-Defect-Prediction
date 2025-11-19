# Usage Guide - Code Defect Analysis

This guide shows you how to use the NASA defect prediction system to analyze your code for defects.

## Quick Start

### 1. Analyze a Single File

```bash
python analyze_code.py path/to/your/file.py --model JM1
```

### 2. Analyze a Directory

```bash
python analyze_code.py path/to/your/directory/ --model JM1
```

### 3. Run Interactive Demo

```bash
python quick_demo.py
```

## Command-Line Options

```bash
python analyze_code.py <path> [options]

Options:
  path                  File or directory to analyze
  --model {CM1,JM1,KC1,KC2,PC1}
                       NASA model to use (default: JM1)
  --simple             Use simple prediction only (faster)
  --output DIR         Output directory for results (default: ./analysis_results)
  --help              Show help message
```

## Examples

### Example 1: Analyze with Different Models

```bash
# Use JM1 model (trained on 10,885 samples, good for general use)
python analyze_code.py mycode.py --model JM1

# Use KC1 model (trained on 2,109 samples)
python analyze_code.py mycode.py --model KC1

# Use CM1 model (trained on 498 samples, smaller dataset)
python analyze_code.py mycode.py --model CM1
```

### Example 2: Analyze Project Directory

```bash
# Analyze all Python files in a project
python analyze_code.py ./my_project --model JM1 --output ./defect_analysis
```

### Example 3: Use in Python Script

```python
from analyze_code import CodeDefectAnalyzer

# Initialize analyzer
analyzer = CodeDefectAnalyzer(model_name='JM1', use_unified_framework=False)

# Analyze a file
result = analyzer.analyze_file('mycode.py')

# Check result
if result['prediction']['is_defective']:
    print(f"⚠️  File may have defects!")
    print(f"   Probability: {result['prediction']['defect_probability']:.2%}")
    print(f"   LOC: {result['metrics']['loc']}")
    print(f"   Complexity: {result['metrics']['cyclomatic_complexity']}")
else:
    print("✓ File looks clean!")

# Analyze directory
results = analyzer.analyze_directory('./my_project')

# Save results
analyzer.save_results(results, output_dir='./analysis_output')
```

## Understanding the Results

### Prediction Status

- **DEFECTIVE**: Code likely contains defects (probability > 50%)
- **CLEAN**: Code appears clean (probability ≤ 50%)

### Key Metrics

| Metric | Description |
|--------|-------------|
| **LOC** | Lines of Code |
| **Cyclomatic Complexity** | Code complexity (higher = more complex) |
| **Max Nesting Depth** | Deepest nesting level |
| **Functions** | Number of functions |
| **Classes** | Number of classes |
| **Loops** | Number of loops |
| **Conditionals** | Number of if statements |

### Defect Probability

- **0-30%**: Low risk - code appears clean
- **30-50%**: Moderate risk - review recommended
- **50-70%**: High risk - likely has defects
- **70-100%**: Very high risk - needs attention

## Output Files

When you run the analyzer, it creates three output files:

### 1. JSON Report (`analysis_TIMESTAMP.json`)

Detailed results in JSON format for programmatic use.

```json
{
  "file_path": "sample.py",
  "model": "JM1",
  "prediction": {
    "is_defective": true,
    "defect_probability": 0.6267,
    "status": "DEFECTIVE"
  },
  "metrics": {
    "loc": 103,
    "cyclomatic_complexity": 18,
    "max_nesting_depth": 14
  }
}
```

### 2. Text Report (`report_TIMESTAMP.txt`)

Human-readable summary report.

```
======================================================================
CODE DEFECT ANALYSIS REPORT
======================================================================
Generated: 2025-11-19 10:35:05
Model Used: JM1
Files Analyzed: 4

SUMMARY
----------------------------------------------------------------------
  Total Files:      4
  Defective:        4 (100.0%)
  Clean:            0 (0.0%)
  Avg Defect Prob:  63.99%
```

### 3. CSV Summary (`summary_TIMESTAMP.csv`)

Spreadsheet-friendly format for further analysis.

```csv
File,Status,Defect Probability,LOC,Complexity,Functions
buggy_processor.py,DEFECTIVE,62.67%,103,18,11
calculator.py,DEFECTIVE,52.44%,32,3,5
```

## Model Selection Guide

### Which Model Should I Use?

| Model | Dataset Size | Best For |
|-------|--------------|----------|
| **JM1** | 10,885 samples | General purpose, most balanced |
| **KC1** | 2,109 samples | Medium-sized projects |
| **KC2** | 522 samples | Specific defect patterns |
| **PC1** | 1,109 samples | Low defect rate scenarios |
| **CM1** | 498 samples | Quick analysis |

**Recommendation**: Start with **JM1** (default) as it's trained on the largest dataset and provides the most reliable predictions.

## Interpreting Results

### High Complexity Warning Signs

If your code shows these characteristics, review it carefully:

- **Cyclomatic Complexity > 10**: Too complex, hard to test
- **Max Nesting Depth > 5**: Deeply nested code, hard to understand
- **LOC > 200 per function**: Function too long, should be split
- **Many loops and conditionals**: Complex logic, prone to bugs

### Common Defect Patterns Detected

The models are trained to detect:

1. **Index errors**: Off-by-one errors, array bounds issues
2. **Null/None handling**: Missing null checks
3. **Resource leaks**: Files/connections not closed
4. **Exception handling**: Bare except blocks, silent failures
5. **Logic errors**: Complex nested conditions
6. **Security issues**: Use of eval, exec, etc.

## Best Practices

### 1. Regular Analysis

```bash
# Add to CI/CD pipeline
python analyze_code.py src/ --model JM1 --output ./defect_reports
```

### 2. Focus on High-Risk Files

Review files with:
- Defect probability > 70%
- High cyclomatic complexity (> 15)
- Deep nesting (> 10 levels)

### 3. Track Improvements

```bash
# Before refactoring
python analyze_code.py mycode.py > before.txt

# After refactoring
python analyze_code.py mycode.py > after.txt

# Compare results
diff before.txt after.txt
```

### 4. Combine with Code Review

Use defect prediction to:
- Prioritize code review efforts
- Identify modules needing more testing
- Guide refactoring decisions

## Sample Analysis Workflow

```bash
# Step 1: Analyze your codebase
python analyze_code.py ./src --model JM1 --output ./analysis

# Step 2: Review the report
cat ./analysis/report_*.txt

# Step 3: Check the CSV for sorting/filtering
# Open summary_*.csv in Excel or similar

# Step 4: Focus on high-risk files
# Review files with probability > 70%

# Step 5: After fixes, re-analyze
python analyze_code.py ./src --model JM1 --output ./analysis_after
```

## Troubleshooting

### Error: "Model file not found"

```bash
# Make sure you've trained the models first
python train_nasa_models.py
```

### Error: "Module not found"

```bash
# Install dependencies
pip install numpy pandas scikit-learn scipy imbalanced-learn
```

### Low Accuracy

- The models are trained on NASA C/C++ metrics
- Python code may have different characteristics
- Use as a guide, not absolute truth
- Combine with other tools and manual review

## Advanced Usage

### Batch Processing

```python
import os
from analyze_code import CodeDefectAnalyzer

analyzer = CodeDefectAnalyzer(model_name='JM1')
all_results = []

# Process multiple directories
for project_dir in ['project1', 'project2', 'project3']:
    results = analyzer.analyze_directory(project_dir)
    all_results.extend(results)

# Generate combined report
analyzer.save_results(all_results, './combined_analysis')
```

### Custom Thresholds

```python
# Set custom defect threshold
HIGH_RISK_THRESHOLD = 0.8

for result in results:
    if result['prediction']['defect_probability'] > HIGH_RISK_THRESHOLD:
        print(f"⚠️  CRITICAL: {result['file_path']}")
        print(f"   Probability: {result['prediction']['defect_probability']:.2%}")
```

### Integration with Testing

```python
import unittest
from analyze_code import CodeDefectAnalyzer

class TestCodeQuality(unittest.TestCase):
    def test_no_high_risk_defects(self):
        analyzer = CodeDefectAnalyzer(model_name='JM1')
        results = analyzer.analyze_directory('./src')

        high_risk = [r for r in results
                     if r['prediction']['defect_probability'] > 0.7]

        self.assertEqual(len(high_risk), 0,
                        f"Found {len(high_risk)} high-risk files")
```

## Support

- See `NASA_DATASET_README.md` for dataset details
- See `README.md` for project overview
- Check sample code in `sample_code/` directory
- Run `quick_demo.py` for interactive examples

## Performance Notes

- Analysis time: ~0.1-0.5 seconds per file
- Memory usage: ~50-100MB for model loading
- Suitable for CI/CD integration
- Can process hundreds of files in minutes
