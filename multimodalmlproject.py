import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

try:
    from xgboost import XGBClassifier
    IS_BOOST_READY = True
except Exception:
    IS_BOOST_READY = False

PRIMARY_DATASET_LOC = "DataSet.csv"
RESPONSE_VARIABLE = "TARGET_COL"

HIT_LABEL = "1"
MISS_LABEL = "0"

THRESHOLD_FEATS = 12
THRESHOLD_RECORDS = 500

SPLIT_RATIO = 0.2
UNIFORM_SEED = 42

NB_VARIANT = "gaussian"
