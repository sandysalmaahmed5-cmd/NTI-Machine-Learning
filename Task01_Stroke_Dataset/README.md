# Stroke Prediction

**Task type:** classification  |  **Target:** `stroke`

## Steps (see `notebook.ipynb`)
1. Load and understand the data
2. Cleaning (missing values, duplicates, outliers) and EDA
3. Encoding and scaling, train/test split
4. Train and compare all the models
5. Pick the best model with cross validation, tune it with GridSearchCV
6. Final evaluation on the test set (Accuracy, Precision, Recall, F1, ROC-AUC)

## Run the app
1. Run `notebook.ipynb` to the end (it creates `model.joblib` in this folder).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

Live app: (add the link here)
