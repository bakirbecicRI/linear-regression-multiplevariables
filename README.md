# Multiple Linear Regression (NumPy + Pandas)

This project demonstrates how to implement multiple linear regression **from scratch**, without using machine learning libraries like scikit-learn. The model is built using `NumPy` for numerical operations and `Pandas` for data handling.

## Contents

- Loading and processing data from a CSV file  
- Feature normalization (feature scaling)  
- Implementation of the cost function  
- Implementation of gradient descent  
- Training the model  
- Predicting house prices based on input features  

## Dataset

The input CSV file (`data.csv`) contains housing data with the following columns:

- `size`: square footage of the house  
- `rooms`: number of rooms  
- `price`: price of the house  

## How to Run

1. Install the required libraries (if not already installed):

```bash
pip install numpy pandas
```

2. Run the script:

```bash
python main.py
```

## Example Prediction Input

The model predicts the price of a house based on two features:
- size in square feet (e.g., 1650)
- number of rooms (e.g., 3)

Example output:
```
Predicted house price: $293081.46
```

## Notes

The model uses gradient descent for training. Input normalization is required due to the different scales of the features. Predictions are returned as floating-point numbers. The model does not include regularization and is intended for educational purposes.

## Author

Project created as part of hands-on learning in linear regression and NumPy-based machine learning (Andrew Ng Course).
