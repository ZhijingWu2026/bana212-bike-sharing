# Hourly Bike-Sharing Demand — BANA 212

An English Streamlit app accompanying the team's simplified notebook. It predicts system-wide hourly rental events from calendar and weather inputs, using a historical-average baseline, Linear Regression, and Random Forest.

## Run locally

Use Python 3.13 (the models were built with Python 3.13.14).

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

All data and trained models are included. The app loads `models.joblib` and never retrains. No API key or external data service is required.

## Deploy to Streamlit Community Cloud

1. Upload the complete contents of this directory to a GitHub repository, keeping the files together at the repository root.
2. At https://share.streamlit.io/ choose Create app, then deploy from the repository.
3. Select the repository, its branch, and `app.py` as the entry point.
4. In Advanced settings, select Python **3.13** to match the model environment.
5. Deploy, then test the resulting public URL in an incognito/private browser window.
6. Submit the public app URL and GitHub repository URL. A localhost URL does not satisfy the public-deployment requirement.

Official deployment documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## App sections

- **Executive summary:** data, business question, approach, results and limits.
- **Descriptive analytics:** hourly, monthly and weather charts, each with explanatory captions.
- **Model performance:** chronological split, test metrics, error comparison, actual-versus-predicted plot, high-demand diagnostics and selected model settings.
- **Interactive prediction:** choose a saved model and use a recorded case, change its hour, or construct a custom scenario. Custom calendar variables come from the chosen date, while unspecified numeric weather variables use pre-test means.

## Demo in class

Open Interactive prediction and choose Random Forest.

1. Recorded demo: October 1, 2012, 08:00. Predicted rentals: about **621.5**. Actual: **784**.
2. Change the hour: 14:00, keeping the other model inputs fixed. Predicted rentals: about **114.1**. This is hypothetical, so no actual outcome is shown.
3. Optionally choose another model. Historical Average uses only hour and working-day status, so weather changes do not affect it.

## Results and reproducibility

The model settings and results match `notebook.ipynb`:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Historical Average | 79.95 | 119.59 | 0.65 |
| Linear Regression | 88.94 | 121.36 | 0.64 |
| Random Forest | 66.19 | 95.22 | 0.78 |

Training: Jan 2011–Jun 2012. Validation: Jul–Sep 2012. Test: Oct–Dec 2012. Validation compares forest depths 10 and unrestricted, then final models refit on all pre-test records. The encoder retains categories learned from training only. The forest uses 100 trees, minimum leaf size 2, and random seed 42.

`train_models.py` executes the notebook code offline and saves the fitted objects and supporting results. Run it only when deliberately rebuilding the artifacts:

```bash
python train_models.py
```

In Colab, upload `hour.csv` as `hour.csv` beside the notebook before running it. Pinned dependency versions are necessary to load scikit-learn model artifacts consistently. Only load the trusted joblib file distributed with this repository.

## Files

- `app.py`: Streamlit interface and inference only.
- `notebook.ipynb`: analysis with saved outputs.
- `train_models.py`: offline training and export.
- `models.joblib`: encoder, fitted regression models, saved historical means and defaults.
- `metadata.json`: model settings, reference predictions, versions and source hashes.
- `hour.csv`: UCI hourly source data.
- Other `*.csv`: notebook-derived evaluation outputs.
- `requirements.txt`: pinned environment.
- `commuter_scene.png`: Original minimalist bicycle and Washington, D.C. illustration from the presentation (AI-generated).

## Limitations

This is a historical study of one city, not a present-day forecast. Weather inputs are observed; real advance forecasts would need forecast weather and evaluation. There are 165 unrecorded hours and only three severe-weather records. Counts do not establish station-level inventory or repositioning needs. An earlier exploratory version examined the test quarter, so the evaluation is not a fresh independent confirmation. High-demand test hours still show substantial underprediction. Hypothetical input changes do not measure causal effects; custom temperature and mean apparent temperature can form unusual combinations.

## Attribution

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. The included hourly file contains 17,379 rows; counts in this app come from that file. Aggregate tables are derived from the analysis. The sidebar illustration is AI-generated and does not depict a documented historical observation.

Team: Donya Bagheri, Chia-Chi Chang, Siyi Li, Zhijing Wu.
