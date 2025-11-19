# Software Defect Prediction

Unified ML-Graph Framework for Automated Software Defect Mitigation with NASA Promise Dataset Integration

## Overview

This project implements a comprehensive software defect prediction system with three main phases:

1. **Defect Prediction**: Ensemble ML models for predicting defective code modules
2. **Defect Localization**: Graph Attention Networks for pinpointing defective code locations
3. **Bug Fix Generation**: Automated bug fix suggestions

## Features

- **NASA Promise Dataset Support**: Pre-trained models on 5 NASA datasets (CM1, JM1, KC1, KC2, PC1)
- **Ensemble Learning**: Combines Random Forest, SVM, and Decision Trees
- **SMOTE-Tomek**: Handles imbalanced datasets
- **Feature Selection**: Mutual information-based feature selection
- **Cross-Validation**: Robust model evaluation

## Quick Start

### Train Models on NASA Datasets

```bash
python train_nasa_models.py
```

### Use Pre-trained Models

```python
from defect_prediction import load_nasa_model

model = load_nasa_model('JM1')
predictions, probabilities = model.predict(X)
```

### Run Demo

```bash
python demo_nasa_models.py
```

## Documentation

See [NASA_DATASET_README.md](NASA_DATASET_README.md) for detailed documentation on NASA dataset integration.

## Project Structure

- `defect_prediction.py` - Core defect prediction module
- `nasa_dataset_loader.py` - NASA dataset loader
- `train_nasa_models.py` - Training script
- `demo_nasa_models.py` - Demo and testing script
- `unified_framework.py` - Integrated framework
- `models/` - Trained models and results
