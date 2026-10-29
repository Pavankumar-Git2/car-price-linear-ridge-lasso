# Car Price Prediction — Linear, Ridge and Lasso

An assignment comparing three regression models using the [HellBuoy car-price dataset on Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction). Target: `price`. Assignment due: **November 1, 2026**.

## Test comparison

All metrics use the same 41 held-out cars. MAE/RMSE use original price units; MSE uses squared price units; R² is unitless.

| Model | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| Linear Regression | 1,763.57 | 7,128,546.78 | 2,669.93 | 0.9097 |
| Ridge | 1,860.67 | 7,515,138.98 | 2,741.38 | 0.9048 |
| Lasso | 1,962.95 | 8,357,517.65 | 2,890.94 | 0.8941 |

## Three-sentence conclusion

Ridge is the preferred generalization candidate from training-only cross-validation (RMSE 2,461.86 versus 2,887.98 for Linear Regression), although Linear Regression wins this 41-car test split (RMSE 2,669.93, R² = 0.9097). Regularization controls coefficient size: Ridge has L2 coefficient norm 17,031.0 versus 23,191.7 for unregularized regression, while Lasso sets 15 of 63 coefficients to zero. Regularization reduces model flexibility and sensitivity to correlated predictors, but the disagreement between cross-validation and the small test set means its generalization advantage remains uncertain.

## Method

- 205 cars; no missing cells or exact duplicate rows.
- Fixed 80/20 train/test split with seed 42: 164 training cars and 41 test cars.
- Drop `car_ID`; replace full `CarName` with a normalized brand. Treat the ordered `symboling` risk score as numeric.
- Numeric median imputation and standardization; categorical mode imputation and one-hot encoding with a reference category. All learned preprocessing stays inside the cross-validation pipeline.
- Five-fold shuffled training-only cross-validation, seed 42, selects alpha by RMSE. Ridge alpha: **0.31622777**; Lasso alpha: **6.81292069**. The held-out set is not used for tuning.
- OLS uses no penalty, Ridge uses L2 and Lasso uses L1. Numeric coefficients are per training standard deviation; dummy coefficients are category-versus-reference effects. Their magnitudes should not be ranked together as feature importance.
- CV fold variability is descriptive, not a confidence interval; the tuning CV scores can be optimistic. Predictions are not clipped; these unconstrained models produce some negative predictions, which remain in the evaluation.

## Run the notebook

Use Python 3.11+ (tested with Python 3.14):

```sh
python -m pip install -r requirements.txt
jupyter notebook Car_Price_Linear_Ridge_Lasso.ipynb
```

Choose **Restart Kernel and Run All Cells**. To execute and save it from a terminal:

```sh
python -m jupyter nbconvert --execute --to notebook --inplace Car_Price_Linear_Ridge_Lasso.ipynb --ExecutePreprocessor.timeout=300
```

The notebook uses `CarPrice_Assignment.csv` in the current directory or `data/`, or an attached Kaggle dataset. Otherwise it downloads the original source archive (internet required). On Kaggle, attach **hellbuoy/car-price-prediction** before running; the notebook locates the CSV under `/kaggle/input` and does not need internet in that case. A SHA-256 check pins the exact source snapshot.

## Files

- `Car_Price_Linear_Ridge_Lasso.ipynb`: executed notebook, all three models, tables, figures and conclusion.
- `results/model_comparison.csv`: all four requested test metrics.
- `results/coefficients.csv`: all 63 coefficients across the three models.
- `results/coefficient_summary.csv`: coefficient norms, zero counts and intercepts.
- `results/cross_validation.csv`: alpha selections and CV scores.
- `results/train_test_rmse.csv`, `test_predictions.csv`, `split_manifest.csv`: evaluation audit trail.
- `results/CONCLUSION.md`: the three-sentence conclusion.
- `results/*.png`: notebook figures.
- `results/provenance.json`: dataset hash, split parameters and software versions.

## Source and interpretation

Dataset: [Car Price Prediction Multiple Linear Regression, HellBuoy](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction). The original archive includes a CSV and data dictionary; the dictionary cites the UCI Automobile dataset. The Kaggle API lists its license as unknown, so this repository links to the source rather than republishing the raw CSV. Price units are retained without assuming a currency.

The dataset is small and historical. These results do not establish performance on current market prices, future periods or new brands. Correlated features can destabilize individual coefficients; they are conditional associations, not causal effects.

References: [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
