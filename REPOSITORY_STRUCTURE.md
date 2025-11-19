# Clean Repository Structure

## 📁 Files (8 total)

### Core Simplified Framework (5 files)
1. **phase1_prediction.py** (9.4 KB)
   - ML-based Defect Prediction
   - Ensemble: Random Forest + SVM + Decision Tree
   - SMOTE-TOMEK balancing
   - Mutual Information feature selection

2. **phase2_localization.py** (12 KB)
   - GAT-based Defect Localization
   - AST to Graph conversion
   - 2-layer Graph Attention Network
   - Top-N suspicious node identification

3. **phase3_bug_fix.py** (17 KB)
   - RATG (Retrieval-Augmented Template Generation)
   - CodeBERT-based retrieval
   - Template database with 8 bug patterns
   - Fix validation

4. **main_framework.py** (13 KB)
   - Unified integration of all 3 phases
   - Complete pipeline: Prediction → Localization → Fixing
   - JSON + Text report generation

5. **SIMPLE_IMPLEMENTATION_GUIDE.md** (12 KB)
   - Comprehensive usage guide
   - Implementation details
   - Screenshots checklist
   - Troubleshooting

### Dependencies (3 files)
6. **defect_prediction.py**
   - Software metrics extraction
   - Used by phase1_prediction.py

7. **nasa_dataset_loader.py**
   - Loads NASA Promise datasets (ARFF format)
   - Automatic download and caching

8. **README.md**
   - Main project documentation

---

## 📂 Directories (3 total)

### nasa_datasets/
- CM1.arff (498 samples, 57 KB)
- JM1.arff (10,885 samples, 853 KB)
- KC1.arff (2,109 samples, 166 KB)
- KC2.arff (522 samples, 55 KB)
- PC1.arff (1,109 samples, 105 KB)

**Total**: 15,123 samples, 5 datasets

### models/
- trained_models/
  - CM1_model.pkl (837 KB)
  - JM1_model.pkl (7.2 MB)
  - KC1_model.pkl (2.8 MB)
  - KC2_model.pkl (1.1 MB)
  - PC1_model.pkl (1.5 MB)
- results/
  - training_report.txt
  - training_results.json
  - training_summary.csv

**Total**: 14 MB of trained models

### framework_results/
- results.json (9.3 KB)
- report.txt (1.4 KB)

Test execution results from main_framework.py

---

## 🗑️ Removed Files (29 total)

### Old Implementation Files (9 files)
- analyze_code.py
- bug_fix.py
- complete_pipeline_demo.py
- defect_localization.py
- demo_nasa_models.py
- enhanced_training.py
- quick_demo.py
- train_nasa_models.py
- unified_framework.py

### Redundant Documentation (5 files)
- COMPREHENSIVE_RESULTS.md
- NASA_DATASET_README.md
- RESULTS_SUMMARY.md
- USAGE_GUIDE.md
- SIMPLIFIED_FRAMEWORK_COMPLETE.md

### Directories Removed (3 directories)
- analysis_results/ (3 files)
- enhanced_models/ (8 files)
- sample_code/ (4 files)

**Total Deleted**: 5,957 lines of code removed

---

## 🎯 Repository Status

**Clean and Focused**: Repository now contains only the essential simplified framework files as requested.

**Quick Start**:
```bash
# Run the complete framework
python main_framework.py

# View implementation guide
cat SIMPLE_IMPLEMENTATION_GUIDE.md
```

**Structure**:
```
Software-Defect-Prediction/
├── phase1_prediction.py           # Phase 1: Defect Prediction
├── phase2_localization.py         # Phase 2: Defect Localization
├── phase3_bug_fix.py              # Phase 3: Bug Fix Generation
├── main_framework.py              # Unified Framework
├── SIMPLE_IMPLEMENTATION_GUIDE.md # Complete Guide
├── defect_prediction.py           # Dependency
├── nasa_dataset_loader.py         # Dependency
├── README.md                      # Main Docs
├── nasa_datasets/                 # Datasets (5 files)
├── models/                        # Trained Models (5 files + results)
└── framework_results/             # Test Results (2 files)
```

---

*Repository cleaned: 2025-11-19*
*Commit: d85bb82*
*Branch: claude/nasa-dataset-model-training-01XEzNE1kvx1vMHdBAbE7o6M*
