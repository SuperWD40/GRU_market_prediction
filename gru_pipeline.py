import numpy as np
import pandas as pd
import time
from datetime import timedelta, datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import json

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2, output_size=1, dropout=0.0, device="cpu"):
        super(GRUModel, self).__init__()
        self.device = device
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, device=self.device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, device=self.device)
    
    def forward(self, x):
        out, _ = self.gru(x.to(self.device))
        out = self.fc(out[:, -1, :])
        return out

class Pipeline:
    def __init__(self, model, dataset, inputs, outputs, ticker=None, freq=None):
        self.id = uuid.uuid4()
        self.model = model
        self.dataset = dataset[inputs + [outputs]]
        self.device = self.model.device
        self.inputs = inputs
        self.outputs = outputs
        self.ticker = ticker
        self.freq = freq

    def hyper_param(self, lr=0.001, batch_size=32, num_workers=4, epochs=10, delta=1, weight_decay=1e-4):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta
        self.num_workers=num_workers
        self.weight_decay=weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.HuberLoss(delta=self.delta)

    def preprocess(self, train_size=0.75, val_size=0.125, test_size=0.125, seq_size=10):
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Les proportions train, val, test doivent avoir une somme de 1.0")
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seq_size = seq_size
    
        # Convertir les données en tenseurs PyTorch
        self.inputs_scaler = MinMaxScaler()
        scaled_inputs = self.dataset[self.inputs].values
        scaled_inputs = self.inputs_scaler.fit_transform(scaled_inputs)
        features = torch.tensor(scaled_inputs, dtype=torch.float32)
        
        self.outputs_scaler = MinMaxScaler()
        scaled_outputs = self.dataset[self.outputs].values.reshape(-1, 1)
        scaled_outputs = self.outputs_scaler.fit_transform(scaled_outputs).reshape(-1)
        labels = torch.tensor(scaled_outputs, dtype=torch.float32)

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

    def train(self):
        self.time_launch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_model = time.time()
        self.train_loss_df = []
        self.val_loss_df = []

        for epoch in range(self.epochs):
            start_epoch = time.time()
            
            # TRAIN
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
            train_loss /= len(self.train_loader)
            self.train_loss = (self.outputs_scaler.inverse_transform(np.array(train_loss).reshape(1, -1))).item()
            
            # VAL
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    val_loss += self.criterion(outputs.squeeze(), y_batch).item()
            val_loss /= len(self.val_loader)
            self.val_loss = (self.outputs_scaler.inverse_transform(np.array(val_loss).reshape(1, -1))).item()

            # TIME
            elapsed_time = time.time() - start_epoch
            elapsed_time_str = str(timedelta(seconds=elapsed_time))
            self.time_elapsed =  round(time.time() - start_model, 4)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {self.train_loss:,.6f}, Validation Loss: { self.val_loss:,.6f}, Time: {elapsed_time_str}")
            self.train_loss_df.append(self.train_loss)
            self.val_loss_df.append(self.val_loss)
        
        # TEST
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
        test_loss /= len(self.test_loader)
        self.test_loss = (self.outputs_scaler.inverse_transform(np.array(test_loss).reshape(1, -1))).item()

        self.y_pred = self.outputs_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
        self.y_test = self.outputs_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

    def loss(self, plot=False):
        if plot:
            plt.figure(figsize=(8,6))
            plt.plot(self.train_loss_df, label="Train Loss")
            plt.plot(self.val_loss_df, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Évolution des pertes (Overfitting si écart grand)")
            plt.show()

        else:
            loss_df = pd.DataFrame({
                "Train loss" : self.train_loss_df,
                'Val loss' : self.val_loss_df
            })
            return loss_df
    
    def pred(self, plot=False):
        if plot:
            plt.figure(figsize=(6,6))
            sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.5)
            plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color="red", linestyle="--")
            plt.xlabel("Actuals")
            plt.ylabel("Predictions")
            plt.title("Predicted vs Actual")
            plt.show()

        else:
            return pd.DataFrame({
                "y_test" : self.y_test,
                "y_pred" : self.y_pred
            })

    def eval(self):
        self.r2 = r2_score(self.y_test, self.y_pred)
        n = len(self.y_test)
        p = next(iter(self.test_loader))[0].shape[-1]
        self.r2_bar = 1 - ((1 - self.r2) * (n - 1)) / (n - p - 1)
        self.mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred) ** 0.5
        self.rmse = self.mse ** 0.5
        
        print(f"Time elapsed: {self.time_elapsed}")
        print(f"Train Loss: {(self.train_loss):,.6f}")
        print(f"Val Loss: {(self.val_loss):,.6f}")
        print(f"Test Loss: {(self.test_loss):,.6f}")
        print(f"R^2 : {self.r2:.4f}")
        print(f"R^2 ajusté : {self.r2_bar:.4f}")
        print(f"MAPE : {self.mape:.4f}")
        print(f"MSE : {self.mse:,.6f}")
        print(f"RMSE : {self.rmse:,.6f}")

    def metadata(self):
        return {
            'General': {
                'Ticker': self.ticker,
                'ID': str(self.id),
                'Window size': self.seq_size,
                'Training time': self.time_elapsed,
                'Training device': self.device,
            },
            'Dataset': {
                'Start date': str(self.dataset.index[0]),
                'End date': str(self.dataset.index[-1]),
                'Dataset Size': len(self.dataset),
                'Dataset freq': self.freq,
                'Train size': self.train_size,
                'Val size': self.val_size,
                'Test size': self.test_size,
            },
            'Model': {
                'Input size': len(self.inputs),
                'Output size': len(self.outputs),
                'Hidden size': self.model.hidden_size,
                'Num layers': self.model.num_layers,
                'Dropout': self.model.dropout,
            },
            'Hyperparameter': {
                'Batch size': self.batch_size,
                'Epochs': self.epochs,
                'Num workers': self.num_workers,
                'Learning rate': self.lr,
                'Delta': self.delta,
                'Weight decay': self.weight_decay
            },
            'Model Performance': {
                'Train Loss': round(self.train_loss, 6),
                'Val Loss': round(self.val_loss, 6),
                'Test Loss': round(self.test_loss, 6),
                'R2': round(self.r2, 4),
                'R2_bar': round(self.r2_bar, 4),
                'MAPE': round(self.mape, 4),
                'MSE': round(self.mse, 4),
                'RMSE': round(self.rmse, 4),
            },
            'Inputs and Outputs': {
                'Inputs': self.inputs,
                'Outputs': self.outputs
            },
        }

