import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, make_scorer

def print_scores(X, y, y_pred):
    # Calculate test metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    r2_adj = adjusted_r_squared(r2, X)
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Adj. R^2 Score: {r2_adj:.2f}")

# Define function for calculating adjusted r-squared
def adjusted_r_squared(r2, X):
    adjusted_r2 = 1 - ((1 - r2) * (len(X) - 1) / (len(X) - X.shape[1] - 1))
    return adjusted_r2 

def rmse_scorer(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))



