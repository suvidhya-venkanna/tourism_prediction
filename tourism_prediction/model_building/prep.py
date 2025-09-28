# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths


#os.environ["HF_TOKEN"] = "  "  # please use your token
api = HfApi(token=os.getenv("HF_TOKEN"))


# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/Suvidhya/tourism-package-prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Encoding the categorical 'Type' column
# The 'Type' column does not exist in this dataset.
# I will remove this line as it's causing an error.
# label_encoder = LabelEncoder()
# df['Type'] = label_encoder.fit_transform(df['Type'])

target_col = 'ProdTaken' # Correct target column based on the data description


# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="Suvidhya/tourism-package-prediction",

        repo_type="dataset",
    )
