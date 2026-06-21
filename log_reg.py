"""
Logistic Regression baseline on the CROMA joint embeddings (per-pixel).

Same purpose and SAME data as rand_forest.py: it reuses the exact split 
and the exact FID draws of the learning curve, so the Logistic
Regression and the Random Forest are point-by-point comparable.

Simple case for now: default regularization (C=1.0), no tuning. Logistic
Regression is scale-sensitive, so features are standardized first.
TODO: tune C with cross-validation INSIDE the training data (LogisticRegressionCV)
      without carving out a separate val set, to keep it comparable to the RF.
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from rand_forest import signal, learning_curve


def make_lr():
    """Standardize the 768 features, then logistic regression with default C=1.0."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,                      # default regularization (simple case)
            class_weight="balanced",    # unbalanced classes (lots of background)
            max_iter=1000,              # give lbfgs room to converge on 768 features
            random_state=0,
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
