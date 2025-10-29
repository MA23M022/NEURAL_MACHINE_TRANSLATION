from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from backend.model.encoder_decoder_lstm import MyModel
from backend.model.evaluate_lstm import evaluate
from backend.config.prepare_dataloader import get_dataloader
from backend.config.training_epoch import train_epoch
from backend.utils.util import timeSince
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train(train_dataloader, model, n_epochs, learning_rate=0.001):
    start = time.time()
    losses_list = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        losses_list.append(loss)
        print(f"At the {epoch}-th epoch :-")
        print('%s %.4f' % (timeSince(start, epoch / n_epochs), loss))

    return losses_list


def evaluateBLEU(model, pairs, input_lang, output_lang, n=100):
    smoothie = SmoothingFunction().method4
    total_score = 0

    for i in range(n):
        pair = random.choice(pairs)
        output_words, _ = evaluate(model, pair[0], input_lang, output_lang)

        # Remove <EOS> and other special tokens from predicted output
        output_words = [word for word in output_words if word not in ['<EOS>', '<PAD>', '<SOS>']]

        # Prepare reference (true translation) token list
        reference = pair[1].split()  # ground truth
        candidate = output_words     # model output

        # Calculate BLEU score for the current pair
        score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
        total_score += score

        print(f"Input: {pair[0]}")
        print(f"Target: {pair[1]}")
        print(f"Predicted: {' '.join(candidate)}")
        print(f"BLEU: {score:.4f}\n")

    avg_bleu = total_score / n
    print(f"Average BLEU score over {n} samples: {avg_bleu:.4f}")
    return avg_bleu


def log_and_train(data_frame : pd.DataFrame, hidden_size : int, batch_size : int, n_epochs : int,
                                learning_rate : float, n_samples : int = 50, run_name : str = "Lstm_model"):
    
    input_lang, output_lang, train_dataloader, pairs = get_dataloader("bn", "en", data_frame, batch_size = batch_size)

    model = MyModel(input_lang.n_words, output_lang.n_words, hidden_size).to(device)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Machine Tranlation Models")

    with mlflow.start_run(run_name = run_name):
        # Log the model hyper parameters
        mlflow.log_param("Input vocan size", input_lang.n_words)
        mlflow.log_param("Output vocab size", output_lang.n_words)
        mlflow.log_param("Embedding size", hidden_size)
        mlflow.log_param("Hidden state size", hidden_size)
        mlflow.log_param("Num_epochs", n_epochs)
        mlflow.log_param("Learning rate", learning_rate)
        mlflow.log_param("Num_samples", n_samples)

        model.train()
        loss_list = train(train_dataloader, model, n_epochs, learning_rate)
        print(f"The loss list is : {loss_list}")
        

        avg_bleu_score = evaluateBLEU(model, pairs, input_lang, output_lang, n=n_samples)
        print(f"The average bleu score is {avg_bleu_score}")

        # Log the metrices
        mlflow.log_metric("Final training loss", loss_list[-1])
        mlflow.log_metric("Average Bleu score", avg_bleu_score)

        # Log the model
        signature = None
        for data in train_dataloader:
            input_tensor, _ = data
            model.eval()
            with torch.no_grad():
                decoder_outputs, _, _ = model(input_tensor)
                signature = infer_signature(input_tensor, decoder_outputs)

            break

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
        


if __name__ == "__main__":
    df = pd.read_csv("backend/data/cleaned_data/cleaned_eng_bng.csv", encoding='utf-8')
    hidden_size = 128
    batch_size = 32
    n_epochs = 1
    learning_rate = 0.001
    n_samples = 50

    log_and_train(df, hidden_size, batch_size, n_epochs, learning_rate,
                   n_samples = n_samples, run_name = "Bidirectional_Lstm_model")

    print("Model training has been done")

    


