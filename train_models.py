"""Offline preparation only. Execute the submitted notebook, then save its fitted models."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import platform
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
notebook = json.loads((ROOT / 'notebook.ipynb').read_text())
namespace = {}
# The notebook expects hour.csv in its working directory.
os.chdir(ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'notebook.ipynb', 'exec'), namespace)
            plt.close('all')
n = namespace
bundle = {
    'encoder': n['encoder'], 'Linear Regression': n['lr'], 'Random Forest': n['rf'],
    'average': n['average'], 'global_mean': float(n['y_past'].mean()),
    'cat_cols': n['cat_cols'], 'num_cols': n['num_cols'], 'features': n['features'],
    'numeric_means': n['past'][n['num_cols']].mean().to_dict(),
    'example': n['example'],
}
# Saved artifacts live beside app.py for straightforward deployment.
joblib.dump(bundle, ROOT / 'models.joblib', compress=3)
n['results'].to_csv(ROOT / 'test_results.csv', index=False)
n['validation_table'].to_csv(ROOT / 'validation_results.csv', index=False)
n['peak_results'].to_csv(ROOT / 'peak_results.csv')
pred = n['test'][['timestamp','cnt']].copy()
for name, values in n['predictions'].items():
    pred[name] = values
pred.to_csv(ROOT / 'test_predictions.csv', index=False)
metadata = {
    'python': platform.python_version(), 'sklearn': sklearn.__version__,
    'dataset_sha256': hashlib.sha256((ROOT/'hour.csv').read_bytes()).hexdigest(),
    'notebook_sha256': hashlib.sha256((ROOT/'notebook.ipynb').read_bytes()).hexdigest(),
    'training_rows': len(n['train']), 'validation_rows': len(n['valid']), 'test_rows': len(n['test']),
    'peak_threshold': float(n['peak_threshold']),
    'rmse_improvement': float(n['improvement']),
    'parameters': {
        'Historical Average': {'grouping': ['workingday','hr'], 'fallback': 'pre-test overall mean'},
        'Linear Regression': {'fit_intercept': True, 'negative_predictions': 'clipped to zero', 'tuning': 'none'},
        'Random Forest': {'n_estimators': 100, 'max_depth': n['best_depth'], 'min_samples_leaf': 2, 'random_state': 42},
    },
    'validation_depths': {'10': float(n['validation_errors'][0]), 'None': float(n['validation_errors'][1])},
    'reference_predictions': {'recorded_rf':float(n['rf'].predict(n['example_X'])[0]),
                              'hour14_rf':float(n['rf'].predict(n['scenario_X'])[0])},
}
(ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2))
assert np.isclose(n['rf_rmse'],95.22,atol=.01)
assert np.isclose(metadata['reference_predictions']['recorded_rf'],621.5,atol=.1)
print(n['results'].round(2).to_string(index=False))
print('Saved models and results matching the simplified notebook.')
