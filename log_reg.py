"""
Logistic Regression baseline on the CROMA joint embeddings (per-token).

Same purpose and SAME data as rand_forest.py: it reuses the exact split 
and the exact FID draws of the learning curve, so the Logistic
Regression and the Random Forest are comparable.

Regression is scale-sensitive, so features are standardized first.
tune C with cross-validation using the training data (LogisticRegressionCV)
    without carving out a separate val set, to keep it comparable to the RF.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from rand_forest import signal, learning_curve

#tunning C
def make_lr():
    """Standardize the 768 features and tune C with cross-validation on the training data."""
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 13),   # grid: 1e-3 ... 1e3 (13 values)
            cv=5,                        # 5-fold CV inside the training tokens
            scoring="f1_macro",          # optimize macro-F1
            class_weight="balanced",    # unbalanced classes (lots of background)
            max_iter=2000,              #768 features make the optimization harder, so lbfgs (default) needs more steps to converge
            random_state=0,
            n_jobs=n_jobs,
        ),
    )


if __name__ == "__main__":
    signal("embeddings",
           os.path.expanduser("~/thesis_tiles_120px"),
           "data_shp/label_polygons.shp",
           make_model=make_lr, model_name="LogisticRegression")
    learning_curve("embeddings",
                   os.path.expanduser("~/thesis_tiles_120px"),
                   "data_shp/label_polygons.shp",
                   make_model=make_lr, model_name="LogisticRegression")
