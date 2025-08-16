# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

This project delivers an end-to-end ML pipeline on the classic Heart Disease dataset (UCI). It includes:
- Cleaned dataset with selected features
- Dimensionality reduction (PCA) results
- Supervised & unsupervised models
- Performance evaluation metrics
- Hyperparameter optimization
- Saved model in `.pkl` format
- GitHub-ready repo structure
- Streamlit UI for real-time predictions (Bonus)
- Ngrok instructions to expose the app (Bonus)

## Project Structure
```
heart-ml-pipeline/
├─ data/
│  └─ heart.csv                # auto-downloaded on first run if absent
├─ src/
│  ├─ download_data.py
│  ├─ data_prep.py
│  ├─ train_supervised.py
│  ├─ train_unsupervised.py
│  ├─ evaluate.py
│  └─ pipeline_run.py          # orchestrates everything end-to-end
├─ models/
│  └─ best_model.pkl           # created after training
├─ outputs/
│  └─ plots/                   # confusion matrix, roc curve, pca, clusters
├─ app/
│  └─ streamlit_app.py         # Streamlit UI
├─ requirements.txt
└─ README.md
```

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Get the dataset
The scripts will try to load `data/heart.csv`; if missing, they will attempt to download from known mirrors.
You can also place your own `heart.csv` in `data/` with the standard columns.

## 3) Run the full pipeline
```bash
python -m src.pipeline_run
```
This will:
- Download the dataset if needed
- Clean & preprocess (handle missing/outliers, scale, feature selection)
- PCA visualization
- Train supervised models (LogReg, SVC, RandomForest) with GridSearchCV
- Evaluate on a held-out test set (classification report, ROC-AUC, confusion matrix)
- Train unsupervised KMeans, pick best k by silhouette
- Save the best supervised model to `models/best_model.pkl`
- Save plots to `outputs/plots`

## 4) Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
It loads `models/best_model.pkl` and provides a form to enter features and get real-time predictions.

## 5) (Bonus) Expose the app with Ngrok
Install and authenticate ngrok, then:
```bash
ngrok http 8501
```
Copy the forwarding URL and share it.

## Notes
- Reproducibility is handled via fixed `random_state` values.
- All hyperparameters and model choices can be edited in `train_supervised.py`.
- The Streamlit app reads the feature list from the trained pipeline to avoid drift.
