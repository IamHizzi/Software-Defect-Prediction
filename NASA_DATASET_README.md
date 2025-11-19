# NASA Promise Dataset Implementation

This project includes implementation of software defect prediction models trained on NASA Promise datasets.

## Datasets

The following NASA Promise datasets are supported:

- **CM1**: 498 samples, 21 features, 9.8% defect rate
- **JM1**: 10,885 samples, 21 features, 19.3% defect rate
- **KC1**: 2,109 samples, 21 features, 15.5% defect rate
- **KC2**: 522 samples, 21 features, 20.5% defect rate
- **PC1**: 1,109 samples, 21 features, 6.9% defect rate

## Data Sources

- Primary: [ApoorvaKrisna/NASA-promise-dataset-repository](https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository)
- Fallback: [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository/)

## Project Structure

```
Software-Defect-Prediction/
├── nasa_dataset_loader.py      # Dataset loader for ARFF files
├── train_nasa_models.py        # Training script for NASA datasets
├── demo_nasa_models.py         # Demo script for using pre-trained models
├── defect_prediction.py        # Main defect prediction module (updated)
├── nasa_datasets/              # Cached dataset files
│   ├── CM1.arff
│   ├── JM1.arff
│   ├── KC1.arff
│   ├── KC2.arff
│   └── PC1.arff
└── models/
    ├── trained_models/         # Trained model files
    │   ├── CM1_model.pkl
    │   ├── JM1_model.pkl
    │   ├── KC1_model.pkl
    │   ├── KC2_model.pkl
    │   └── PC1_model.pkl
    └── results/                # Training results
        ├── training_report.txt
        ├── training_results.json
        └── training_summary.csv
```

## Usage

### 1. Training Models on NASA Datasets

To train models on all NASA datasets:

```bash
python train_nasa_models.py
```

This will:
- Download datasets (if not cached)
- Train ensemble models on each dataset
- Perform cross-validation
- Save trained models to `./models/trained_models/`
- Generate training reports in `./models/results/`

### 2. Using Pre-trained Models

#### Load a specific model:

```python
from defect_prediction import load_nasa_model

# Load pre-trained JM1 model
model = load_nasa_model('JM1')

# Make predictions
predictions, probabilities = model.predict(X)

# Evaluate
results = model.evaluate(X_test, y_test)
```

#### List available models:

```python
from defect_prediction import list_available_models

models = list_available_models()
print(f"Available models: {models}")
```

### 3. Running Demo

To see the models in action:

```bash
python demo_nasa_models.py
```

This demonstrates:
- Loading pre-trained models
- Making predictions on test data
- Evaluating model performance
- Testing all available models

### 4. Loading NASA Datasets

To load and explore NASA datasets:

```python
from nasa_dataset_loader import NASADatasetLoader

# Initialize loader
loader = NASADatasetLoader()

# Load a single dataset
X, y, info = loader.load_dataset('JM1')

# Load multiple datasets
datasets = loader.load_all_datasets(['CM1', 'JM1', 'KC1', 'KC2', 'PC1'])
```

## Model Performance

### Training Results Summary

| Dataset | Samples | Features | Accuracy | Precision | Recall | F1-Score | CV F1 Mean |
|---------|---------|----------|----------|-----------|--------|----------|------------|
| CM1     | 498     | 20       | 0.7700   | 0.1905    | 0.4000 | 0.2581   | 0.0819     |
| JM1     | 10,885  | 20       | 0.7235   | 0.3538    | 0.5202 | 0.4212   | 0.1809     |
| KC1     | 2,109   | 20       | 0.8009   | 0.4078    | 0.6462 | 0.5000   | 0.2563     |
| KC2     | 522     | 20       | 0.7905   | 0.5000    | 0.5000 | 0.5000   | 0.4164     |
| PC1     | 1,109   | 20       | 0.8919   | 0.3333    | 0.6000 | 0.4286   | 0.2113     |

**Overall Statistics:**
- Average Accuracy: 0.7954
- Average Precision: 0.3571
- Average Recall: 0.5333
- Average F1-Score: 0.4216

## Model Architecture

Each model uses an ensemble approach with:

1. **Feature Selection**: SelectKBest with mutual information (20 features)
2. **Class Balancing**: SMOTE-Tomek for handling imbalanced classes
3. **Ensemble Voting Classifier**:
   - Random Forest (weight: 2)
   - Support Vector Machine with RBF kernel (weight: 1)
   - Decision Tree (weight: 1)

## Dependencies

```
numpy
pandas
scikit-learn
scipy
imbalanced-learn
```

Install with:
```bash
pip install numpy pandas scikit-learn scipy imbalanced-learn
```

## Dataset Features

The NASA datasets include McCabe and Halstead software metrics:

- Lines of Code (LOC)
- Cyclomatic Complexity (v(g))
- Essential Complexity (ev(g))
- Design Complexity (iv(g))
- Halstead Volume
- Halstead Difficulty
- Halstead Effort
- And 14 more software metrics...

## References

1. Shepperd, M., Song, Q., Sun, Z., & Mair, C. (2013). Data quality: Some comments on the NASA software defect datasets. IEEE Transactions on Software Engineering, 39(9), 1208-1215.

2. [NASA PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository/)

3. [ApoorvaKrisna NASA Dataset Repository](https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository)

## License

NASA Promise datasets are publicly available for research purposes.

## Contact

For issues or questions, please refer to the main project README.
