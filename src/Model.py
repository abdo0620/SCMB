import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"]="."

# Download latest version
path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")

print("Path to dataset files:", path)

