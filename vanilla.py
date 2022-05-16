import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(0)

class VanillaRNN(nn.Module):
    def __init__(self, D, H, O):
        super(VanillaRNN, self).__init__()

        self.D = D
        self.H = H
        self.O = O

        self.Wxh = nn.Parameter(torch.ones((self.D, self.H)).type(torch.DoubleTensor))
        self.Whh = nn.Parameter(torch.ones((self.H, self.H)).type(torch.DoubleTensor))
        self.Who = nn.Parameter(torch.ones((self.H, self.O)).type(torch.DoubleTensor))

    def forward(self, x):
        h = torch.zeros(self.H).type(torch.DoubleTensor)

        N, T, D = x.shape
        for t in range(T):
            x_i = x[:, t, :]
            h = torch.tanh(torch.matmul(x_i, self.Wxh) + torch.matmul(h, self.Whh))

        y_hat = torch.matmul(h, self.Who)
        return y_hat

def plot_model(model, T, csv_fn):
  dataset_test = pd.read_csv(csv_fn)
  real_stock_price = dataset_test.iloc[:, 1:2].values

  # Getting the predicted stock price of 2017
  dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)
  X_test = []

  for i in range(T, inputs.shape[0]):
      X_test.append(torch.tensor(inputs[i-T:i, 0]))

  X_test = torch.stack(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  predicted_stock_price = model(X_test).detach().numpy()
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  # Visualising the results
  plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
  plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
  plt.title('Google Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('Google Stock Price')
  plt.legend()
  plt.show()

if not os.path.exists("Google_Stock_Price_Train.csv"):
    url = "https://raw.githubusercontent.com/kevincwu0/rnn-google-stock-prediction/master/Google_Stock_Price_Train.csv"
    urllib.request.urlretrieve(url, "Google_Stock_Price_Train.csv")

if not os.path.exists("Google_Stock_Price_Test.csv"):
    url = "https://raw.githubusercontent.com/kevincwu0/rnn-google-stock-prediction/master/Google_Stock_Price_Test.csv"
    urllib.request.urlretrieve(url, "Google_Stock_Price_Test.csv")

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values 

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

T = 60
D = 1
H = 8
O = 1

num_pts, _ = training_set_scaled.shape
X_train = []
y_train = []

for i in range(T, num_pts):
  X_train.append(torch.tensor(training_set_scaled[i-T:i]))
  y_train.append(torch.tensor(training_set_scaled[i]))

X_train = np.array(X_train)
y_train = np.array(y_train)

model = VanillaRNN(D, H, O)

loss_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
batch_size = 128
epochs = 2000

losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    idx = np.random.choice(np.arange(X_train.shape[0]), batch_size, replace=False)
    X_sample = torch.stack(list(X_train[idx]))
    y_sample = torch.stack(list(y_train[idx]))

    y_hat = model(X_sample)
    loss = loss_criterion(y_hat, y_sample)
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epochs: {epoch}/{epochs} -- Loss: {loss}")
    losses.append(loss.detach().numpy())

plt.plot(range(len(losses)), losses)
plt.show()

plot_model(model, T, "Google_Stock_Price_Train.csv") # should be nearly perfectly fit
plot_model(model, T, "Google_Stock_Price_Test.csv")