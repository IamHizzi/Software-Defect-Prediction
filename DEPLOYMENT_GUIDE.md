# Streamlit Deployment Guide

## 🚀 Deploy to Streamlit Community Cloud (FREE)

### Step 1: Push Your Code to GitHub

Your code is already in this repository:
```
https://github.com/IamHizzi/Software-Defect-Prediction
```

Branch: `claude/nasa-dataset-model-training-01XEzNE1kvx1vMHdBAbE7o6M`

### Step 2: Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Fill in the details**:
   - **Repository**: `IamHizzi/Software-Defect-Prediction`
   - **Branch**: `claude/nasa-dataset-model-training-01XEzNE1kvx1vMHdBAbE7o6M`
   - **Main file path**: `web_interface.py`

5. **Click "Deploy"**

6. **Wait 2-3 minutes** for deployment to complete

7. **Your app will be live at**:
   ```
   https://iamhizzi-software-defect-prediction-web-interface-xxxxx.streamlit.app
   ```

### Step 3: Share Your Link

Once deployed, you'll get a URL like:
```
https://[your-app-name].streamlit.app
```

You can share this link with anyone!

---

## Alternative: Deploy to Hugging Face Spaces (Also FREE)

### Step 1: Create Hugging Face Account
- Go to: https://huggingface.co/join
- Sign up (free)

### Step 2: Create a New Space
1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Space name**: `software-defect-prediction`
   - **License**: MIT
   - **Space SDK**: Streamlit
   - **Visibility**: Public

### Step 3: Upload Files
1. Clone your space locally or use web interface
2. Upload:
   - `web_interface.py`
   - `main_framework.py`
   - `phase1_prediction.py`
   - `phase2_localization.py`
   - `phase3_bug_fix.py`
   - `defect_prediction.py`
   - `nasa_dataset_loader.py`
   - `requirements.txt`
   - All model files from `models/` directory
   - All dataset files from `nasa_datasets/` directory

### Step 4: Your App Will Be Live At:
```
https://huggingface.co/spaces/[your-username]/software-defect-prediction
```

---

## 📋 Files Already Prepared for Deployment

✅ `web_interface.py` - The Streamlit app
✅ `requirements.txt` - All Python dependencies
✅ `packages.txt` - System dependencies
✅ All framework files (phase1, phase2, phase3, main_framework)
✅ Trained models in `models/` directory
✅ Datasets in `nasa_datasets/` directory

---

## 🔧 Test Locally First

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run web_interface.py
```

Open your browser to: http://localhost:8501

---

## 🎯 Recommended: Streamlit Community Cloud

**Why?**
- Free forever
- Auto-deploys on git push
- Built-in analytics
- Custom domain support
- No server management

**Deployment time**: 2-3 minutes
**Maintenance**: Zero (auto-updates from GitHub)

---

## ⚠️ Important Notes

1. **Large Files**: If deployment fails due to large model files (14 MB total), you may need to:
   - Use Git LFS (Large File Storage)
   - Or regenerate models on first run

2. **First Load**: The app might take 30-60 seconds on first load as it initializes the framework

3. **GitHub Pages Won't Work**: GitHub Pages only hosts static HTML/CSS/JS, not Python applications

---

## 📞 Need Help?

If you encounter issues:
1. Check Streamlit Community Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
2. Or Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces

---

**Ready to deploy? Follow the Streamlit Cloud steps above! 🚀**
