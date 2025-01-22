import numpy as np
import pandas as pd
import time
from datetime import timedelta, datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler # attention pas utilisé pour le moment
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2, output_size=1, dropout=0.0, device="cpu"):
        super(GRUModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, device=self.device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, device=self.device)
    
    def forward(self, x):
        _, hidden = self.gru(x.to(self.device))  # Utiliser uniquement l'état caché final
        out = self.fc(hidden[-1])  # Appliquer le FC sur l'état caché final
        return out

class Pipeline:
    def __init__(self, model, dataset, inputs, outputs):
        self.model = model
        self.dataset = dataset[inputs + [outputs]]
        self.device = self.model.device
        self.inputs = inputs
        self.outputs = outputs

    def hyper_param(self, lr=0.001, batch_size=32, num_workers=4, epochs=10):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers=num_workers #Not used for the moment
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def preprocess(self, train_size=0.75, val_size=0.125, test_size=0.125, seq_size=10):
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Les proportions train, val, test doivent avoir une somme de 1.0")
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seq_size = seq_size
    
        # Convertir les données en tenseurs PyTorch
        features = torch.tensor(self.dataset[self.inputs].values, dtype=torch.float32)
        labels = torch.tensor(self.dataset[self.outputs].values, dtype=torch.float32)

        # Créer les séquences
        num_samples = len(features) - seq_size
        sequences = [features[i:i + seq_size] for i in range(num_samples)]
        sequences = torch.stack(sequences)
        outputss = labels[seq_size:]

        # Créer un TensorDataset avec les séquences
        dataset = TensorDataset(sequences, outputss)

        # Diviser en train, validation et test
        dataset_size = len(dataset)
        train_size = int(train_size * dataset_size)
        val_size = int(val_size * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(random.randint(1, 100))
        )

        # DataLoader pour chaque split
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True
        )
        self.X_test = np.array([data.numpy() for data, _ in test_dataset])
        self.y_test = np.array([label for _, label in test_dataset])

    def train(self):
        self.time_launch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_model = time.time()
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            start_epoch = time.time()
            
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.train_loss = train_loss / len(self.train_loader)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    val_loss += self.criterion(outputs.squeeze(), y_batch).item()
            self.val_loss = train_loss / len(self.val_loader)
            
            elapsed_time = time.time() - start_epoch
            elapsed_time_str = str(timedelta(seconds=elapsed_time))

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {self.train_loss:,.6f}, Validation Loss: { self.val_loss:,.6f}, Time: {elapsed_time_str}")
            self.train_losses.append(self.train_loss)
            self.val_losses.append(self.val_loss)

        self.time_elapsed =  round(time.time() - start_model, 4)

    def loss(self, plot=False):
        if plot:
            plt.figure(figsize=(8,5))
            plt.plot(self.train_losses, label="Train Loss")
            plt.plot(self.val_losses, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Évolution des pertes (Overfitting si écart grand)")
            plt.show()

        else:
            loss_df = pd.DataFrame({
                "Train loss" : self.train_losses,
                'Val loss' : self.val_losses
            })
            return loss_df
    
    def pred(self, plot=False):
        self.model.eval()
        test_loss = 0.0
        y_pred, y_test = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                test_loss += self.criterion(outputs.squeeze(), y_batch).item()
                y_pred.extend(outputs.squeeze().cpu().numpy())
                y_test.extend(y_batch.cpu().numpy())

        if plot:
            plt.figure(figsize=(6,6))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
            plt.xlabel("Actuals")
            plt.ylabel("Predictions")
            plt.title("Predicted vs Actual")
            plt.show()

        else:
            y_df = pd.DataFrame({
                "y_test" : y_test,
                "y_pred" : y_pred
            })
            return y_df


    def eval(self):
        self.model.eval()
        test_loss = 0.0
        y_pred, y_test = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                test_loss += self.criterion(outputs.squeeze(), y_batch).item()
                y_pred.extend(outputs.squeeze().cpu().numpy())
                y_test.extend(y_batch.cpu().numpy())
        self.test_loss = test_loss / len(self.test_loader)
        self.y_test = y_test
        self.y_pred = y_pred
        
        self.r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = self.X_test.shape[1]
        self.r2_bar = 1 - ((1 - self.r2) * (n - 1)) / (n - p - 1)
        self.mape = mean_absolute_percentage_error(y_test, y_pred)
        self.rmse = mean_squared_error(y_test, y_pred) ** 0.5
        
        print(f"Time elapsed: {self.time_elapsed}")
        print(f"Train Loss: {(self.train_loss):,.6f}")
        print(f"Val Loss: {(self.val_loss):,.6f}")
        print(f"Test Loss: {(test_loss):,.6f}")
        print(f"R^2 : {self.r2:.4f}")
        print(f"R^2 ajusté : {self.r2_bar:.4f}")
        print(f"MAPE : {self.mape:.4f}")
        print(f"RMSE : {self.rmse:,.6f}")

    def save(self, path):
        # Charger le fichier CSV s'il existe, sinon créer un DataFrame vide
        if os.path.exists(path):
            results = pd.read_csv(path, index_col=0)
        else:
            results = pd.DataFrame()

        # Construire le dictionnaire des résultats
        results_dict = {
            'Dataset size'  : len(self.dataset),
            'Window size'   : self.seq_size,
            'Hidden size'   : self.model.hidden_size,
            'Num layers'    : self.model.num_layers,
            'Dropout'       : self.model.dropout,
            'Input size'    : len(self.inputs),
            'Train size'    : self.train_size,
            'Val size'      : self.val_size,
            'Test size'     : self.test_size,
            'Batch size'    : self.batch_size,
            'Epochs'        : self.epochs,
            'Learning rate' : self.lr,
            'Train Loss'    : round(self.train_loss, 6),
            'Val Loss'      : round(self.val_loss, 6),
            'Test Loss'     : round(self.test_loss, 6),
            'R2'            : round(self.r2, 4),
            'R2_bar'        : round(self.r2_bar, 4),
            'MAPE'          : round(self.mape, 4),
            'RMSE'          : round(self.rmse, 4),
            'Training time' : self.time_elapsed,
            'Training device': self.device,
            'Outputs'       : self.outputs,
        }

        # Ajouter les entrées
        for n, input_value in enumerate(self.inputs):
            results_dict[f"Input_{n}"] = input_value

        # Convertir en DataFrame (assure l'alignement des colonnes)
        new_entry = pd.Series(results_dict, name=self.time_launch)

        # Concaténer les anciennes et nouvelles données, en alignant sur les colonnes
        results = pd.concat([results, new_entry], axis=1, sort=False)

        # Sauvegarder dans le fichier CSV
        results.to_csv(path)


