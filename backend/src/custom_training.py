import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
from urllib.parse import urlparse

# ------------------ Model Definition ------------------
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Training Function ------------------
def train_and_log(lr, hidden_size, batch_size, run_name="experiment"):
    # 1. Generate dummy dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Model & optimizer
    model = MyModel(input_size=10, hidden_size=hidden_size, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Start MLflow run
    #mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Custom_Neural_Network_Testing")
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("batch_size", batch_size)

        # Training loop
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                _, preds = torch.max(val_outputs, 1)
                val_acc = (preds == y_val).float().mean().item()

            # Log metrics
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 4. Log & Register model

        signature = infer_signature(X_train, model(X_train))
        tracing_scheme = urlparse(mlflow.get_tracking_uri()).scheme
        artifact_path = run_name
        if tracing_scheme != "file":
            mlflow.pytorch.log_model(
                            pytorch_model = model,
                            artifact_path = artifact_path,
                            registered_model_name = run_name,
                            signature = signature
                        )
        else:
            mlflow.pytorch.log_model(
                            pytorch_model = model,
                            artifact_path = artifact_path,
                            signature = signature
                        )


        model_uri = mlflow.get_artifact_uri(artifact_path)

        print(f"Model logged and registered at: {model_uri}")



# ------------------ Main ------------------
if __name__ == "__main__":
    # Run experiments with different hyperparameters
    train_and_log(lr=0.001, hidden_size=20, batch_size=32, run_name="run_1")
    train_and_log(lr=0.0005, hidden_size=50, batch_size=64, run_name="run_2")

