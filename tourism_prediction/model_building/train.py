# ===== IMPORTS =====
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import mlflow
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# ===== MLflow Setup =====
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# ===== Hugging Face Login =====
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not set in environment.")
api = HfApi(token=hf_token)

# ===== Download Dataset from HF =====
dataset_repo_id = "Suvidhya/tourism-package-prediction"
Xtrain_path = hf_hub_download(repo_id=dataset_repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id=dataset_repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=dataset_repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id=dataset_repo_id, filename="ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# ===== Preprocessing =====
numeric_features = ["Age", "NumberOfPersonVisiting", "PreferredPropertyStar",
                    "NumberOfTrips", "MonthlyIncome", "DurationOfPitch",
                    "NumberOfFollowups", "PitchSatisfactionScore",
                    "NumberOfChildrenVisiting"]
categorical_features = ["TypeofContact", "CityTier", "Occupation", "Gender",
                        "MaritalStatus", "Designation", "ProductPitched",
                        "Passport", "OwnCar"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ===== Model Setup =====
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5],
}

pipeline = make_pipeline(preprocessor, xgb_model)

# ===== Train & Log with MLflow =====
with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred_test = best_model.predict(Xtest)
    report = classification_report(ytest, y_pred_test, output_dict=True)
    mlflow.log_metrics({
        "test_accuracy": report['accuracy'],
        "test_recall": report['1']['recall'],
        "test_precision": report['1']['precision'],
        "test_f1-score": report['1']['f1-score']
    })

    # Save locally
    model_file = "best_tourism_model.joblib"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file, artifact_path="model")

# ===== Upload Model to HF =====
model_repo_id = "Suvidhya/tourism-package-model"
repo_type = "model"

# Check if repo exists, else create
try:
    api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=model_repo_id,
    repo_type=repo_type
)

print("Model uploaded to Hugging Face successfully!")
