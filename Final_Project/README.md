# Rain Prediction in Australia (WeatherAUS)

Predict whether it will rain **tomorrow** from today's weather observations.
This is a binary classification problem (`RainTomorrow`: Yes / No).

**Live app:** (add the Streamlit link here)

## Dataset
- WeatherAUS: about 140k daily weather observations from many locations in Australia.
- Target: `RainTomorrow`.
- The data is imbalanced (about 78% No, 22% Yes), so accuracy alone is not enough. Models are compared with F1, Recall, Precision and ROC-AUC.
- Put the csv in `dataset/` (see `dataset/put_csv_here.txt`).

## Project steps (see `notebooks/Weather_Rain_Prediction_Final.ipynb`)
1. **Understanding the data:** features, target, and the task type.
2. **EDA:** target distribution, missing values, outliers, and plots.
3. **Cleaning:** missing values are imputed (Evaporation, Sunshine and the Cloud columns have 40-48% missing, so they were kept, not dropped). Rainfall outliers are real heavy-rain days, so they were kept.
4. **Data leakage:** `RISK_MM` was removed because it contains the next day's rainfall.
5. **Feature engineering:** `Date` was split into `Month` and `Day`.
6. **Preprocessing:** stratified 80/20 split, then a `Pipeline` with `ColumnTransformer` (see the Preprocessing section of the notebook), fitted on the training data only.
7. **Models:** 14 classifiers were trained and compared (Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, SVM (LinearSVC), Naive Bayes, Bagging, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost).
8. **GridSearchCV:** 3-fold cross validation on the training set only. The test set was not used for tuning.
9. **Threshold tuning:** the decision threshold was chosen on a validation split from the training data.
10. **Final evaluation** on the test set, then **deployment**.

## Results
Best model before and after GridSearchCV (F1):

| Model | F1 before | F1 after |
|---|---|---|
| LightGBM | 0.627 | 0.645 |
| XGBoost | 0.622 | 0.641 |
| HistGradientBoosting | 0.626 | 0.635 |
| CatBoost | 0.623 | 0.623 |
| Logistic Regression | 0.599 | 0.599 |

The full tables are in `results/comparison_before_after.csv` and `results/test_results.csv`.

**Final model: LightGBM with threshold 0.32**
- With the default threshold 0.5, recall was low (0.57), so many rainy days were missed.
- With threshold 0.32: **F1 = 0.68, Recall = 0.72, Precision = 0.65, ROC-AUC = 0.90**.
- Top features: Humidity3pm, Sunshine, Pressure and Cloud3pm (`results/feature_importance.png`).
- The model is better at predicting "No rain" than "Rain".

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```
The app loads `models/rain_prediction_pipeline.pkl` (the full pipeline plus the threshold).

> `scikit-learn==1.7.2` is pinned because the model was saved with that version.

## Folder structure
```
Final_Project/
   dataset/        the csv file
   notebooks/      the full notebook
   models/         saved pipeline (.pkl)
   results/        plots and result tables
   app.py          Streamlit app
   requirements.txt
   README.md
```
