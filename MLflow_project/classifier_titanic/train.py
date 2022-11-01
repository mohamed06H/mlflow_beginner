import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
np.random.seed(0)

import mlflow
import mlflow.tensorflow
import warnings

# Load in the data
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# construct the piepline
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Enable auto-logging to MLflow to capture sklearn metrics.
mlflow.sklearn.autolog()

def mlflow_run(params, run_name="Tracking Experiment: titanic - accuracy "):
  with mlflow.start_run(run_name=run_name) as run:
    # get current run and experiment id
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id

    # append classifier to the pipeline
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(C=params))]
        )

    clf.fit(X_train, y_train)

    accuracy_score = clf.score(X_test, y_test)


  return (experimentID, runID)

# Use the model
if __name__ == '__main__':
   # suppress any deprecated warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)

   params = 10
   (exp_id, run_id) = mlflow_run(params)

   print(f"Finished Experiment id={exp_id} and run id = {run_id}")
