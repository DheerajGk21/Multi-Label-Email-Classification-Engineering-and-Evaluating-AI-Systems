import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random

num_folds = 0
seed = 0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 200)

# RandomForest classifier implementation for multi-class classification.
class RandomForest(BaseModel):
    # Initialize the RandomForest classifier
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(
            n_estimators=1000, random_state=seed, class_weight="balanced_subsample"
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        target_names = data.label_encoder.classes_
        classes_in_test = sorted(np.unique(data.y_test))
        target_names_for_report = target_names[classes_in_test]
        print(
            classification_report(
                data.y_test,
                self.predictions,
                target_names=target_names_for_report,
                zero_division=0,
            )
        )

    def data_transform(self) -> None: ...
