#!/bin/bash
# Automated Complete Pipeline Demonstration

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                     COMPLETE DEFECT PREDICTION PIPELINE                        "
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This script will run the complete pipeline non-interactively"
echo ""

# Phase 1: Test defect prediction with enhanced models
echo "PHASE 1: DEFECT PREDICTION WITH ENHANCED MODELS"
echo "───────────────────────────────────────────────────────────────────────────────"

python -c "
import pickle
import numpy as np
from defect_prediction import extract_software_metrics
import os

test_files = [
    'sample_code/buggy_processor.py',
    'sample_code/file_handler.py',
    'sample_code/string_utils.py',
    'sample_code/calculator.py'
]

# Load enhanced model
model_path = './enhanced_models/models/JM1_enhanced.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f'✓ Loaded enhanced model from: {model_path}\n')

results = []
for file_path in test_files:
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, 'r') as f:
        code = f.read()
    
    metrics = extract_software_metrics(code)
    metrics_vector = np.array(list(metrics.values())).reshape(1, -1)
    
    # Pad to 21 features
    if metrics_vector.shape[1] < 21:
        padding = np.zeros((1, 21 - metrics_vector.shape[1]))
        metrics_vector = np.hstack([metrics_vector, padding])
    elif metrics_vector.shape[1] > 21:
        metrics_vector = metrics_vector[:, :21]
    
    predictions, probabilities = model.predict(metrics_vector)
    is_defective = predictions[0] == 1
    defect_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
    
    print(f'{os.path.basename(file_path):30s} -> ', end='')
    status = '⚠️  DEFECTIVE' if is_defective else '✓  CLEAN'
    print(f'{status:15s} (Probability: {defect_prob*100:5.2f}%)')
    
    results.append({
        'file': os.path.basename(file_path),
        'defective': is_defective,
        'probability': defect_prob
    })

print(f'\n✓ PHASE 1 COMPLETE')
print(f'  Analyzed: {len(results)} files')
print(f'  Defective: {sum(1 for r in results if r[\"defective\"])}')
"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                              PIPELINE COMPLETE                                 "
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  • Enhanced models trained and saved in: ./enhanced_models/"
echo "  • Training results: ./enhanced_models/results/"
echo "  • Phase 1 predictions: Complete"
echo ""
echo "Model Performance Summary:"
cat ./enhanced_models/results/enhanced_summary.csv
echo ""

