import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
from urllib.parse import urlparse


input_lang = None
output_lang = None
pairs = None

with open("backend/model/input_lang_obj.pkl", "rb") as f:
    input_lang = pickle.load(f)
with open("backend/model/output_lang_obj.pkl", "rb") as f:
    output_lang = pickle.load(f)
with open("backend/model/language_pairs.pkl", "rb") as f:
    pairs = pickle.load(f)

print(list(input_lang.keys()), input_lang["index2word"][23])
print(list(output_lang.keys()), output_lang["index2word"][23])
print(f"The number of language pairs is {len(pairs)}")

# # 1. Generate dummy dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.long)


# model_v1 = torch.load(f"C:/Users/soura/Desktop/Resume_Projects/NEURAL_MT/mlartifacts/136462932973296277/models/m-8fe5f25ed1734a5296ca1621c4dcb3b0/artifacts/data/model.pth", weights_only=False)

# # model_v1 = mlflow.pytorch.load_model(f"C:/Users/soura/Desktop/Resume_Projects/NEURAL_MT/mlruns/models/run_1")
# m1 = list(model_v1.modules())[0]
# m1.eval()
# with torch.no_grad():
#     print(m1.parameters)
#     val_outputs = m1(X_val)
#     _, preds = torch.max(val_outputs, 1)
#     val_acc = (preds == y_val).float().mean().item()
#     print(f"validation Accuracy {val_acc}")



# # ------------------ Load Registered Model ------------------
# def load_registered_model(run_name, version):
#     model = mlflow.pytorch.load_model(f"models:/{run_name}/{version}")
#     model.eval()
#     print(f"📌 Loaded model version {version} from registry")
#     return model


# if __name__ == "__main__":
#     # Load model version 1 for inference
#     model_v1 = load_registered_model(run_name="run_1", version=1)
#     print(model_v1.parameters())

#     # Load model version 2 for inference
#     model_v2 = load_registered_model(run_name="run_2", version=1)
#     print(model_v2.parameters())

