import pandas as pd
import numpy as np

data_frame=pd.read_csv("data.csv")

X=data_frame[['size', 'rooms']].values
y=data_frame['price'].values
m=len(y)

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X-mu) / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = feature_normalize(X)

#initialization
w = np.zeros(X.shape[1])
b = 0.0

def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X @ w + b
    errors = predictions - y 
    return (1 / (2*m)) * np.dot(errors.T, errors)

def gradient_descent(X, y, w, b, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        predictions = X @ w + b
        errors = predictions - y

        dw = (1 / m)* (X.T @ errors)
        db = (1 / m)* np.sum(errors)

        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iter {i:4}: Cost={cost:.2f}, b={b:.2f}, w={w}")

    return w, b, cost_history

alpha = 0.01
num_iters = 500
w, b, J_history = gradient_descent(X_norm, y, w, b, alpha, num_iters)

print("\nFinalni parametri:")        
print(f"w = {w}")
print(f"b = {b}")

x_input = np.array([1650, 3])
x_input_norm = (x_input - mu) / sigma

predicted_price = x_input_norm @ w + b
print(f"\nPredviđena cijena kuće: ${predicted_price:.2f}")

