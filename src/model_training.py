import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import create_preprocessing_pipeline


def train_model(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = create_preprocessing_pipeline(df, target_column)

    class_weights = {0: 0.624, 1: 2.519}

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weights
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weights
       ),

        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            scale_pos_weight=class_weights[1] / class_weights[0]  
        ),

        "LightGBM": LGBMClassifier(
            random_state=42,
            scale_pos_weight=class_weights[1] / class_weights[0]
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "CatBoost": CatBoostClassifier(
            verbose=0,
            random_state=42,
            class_weights=[class_weights[0], class_weights[1]]
        )
    }

    best_model = None
    best_score = 0
    best_model_name = None

    for name, model in models.items():
        pipeline = ImbPipeline([
            ("preprocess", preprocessor),  
            ("model", model)
        ])
    
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred)
    
        if acc > best_score:
            best_score = acc
            best_model = pipeline
            best_model_name = name

    print(f"\nThe best model: {best_model_name} (Accuracy: {best_score:.4f})")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model