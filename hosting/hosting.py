from huggingface_hub import HfApi
import os

os.environ["HF_TOKEN"] = " "    # please use your token
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",          # local folder containing your app.py and files
    repo_id="Suvidhya/tourism-package-model",          # your Hugging Face Space repo
    repo_type="space",                                 # type is Space
    path_in_repo="",                                   # root of the space
)
