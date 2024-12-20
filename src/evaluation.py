import numpy as np
import pandas as pd
import plotly as plt
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
    return rmse

# Initialize the DataFrame with the first entry (Benchmark model: winner of Zindi challenge)
df_models_rmse = pd.DataFrame([{'Model': 'Benchmark model', 'RMSE': 26.09}])

def print_save_scores(model_name, model, X_train, y_train, y_pred, df):
     # Calculate test metrics
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)
    r2_adj = adjusted_r_squared(r2, X_train)
    
    # Print metrics
    print(f"Model: {model_name}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Adj. R^2 Score: {r2_adj:.2f}")
        
    # Create a new DataFrame to append
    new_row = pd.DataFrame({'Model': [model_name], 'RMSE': [rmse]})
    
    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    return df

# Function to calculate and save RMSE to DataFrame
def save_rmse_from_cv(model_name, y_test, y_test_pred, df):
    # Calculate RMSE using your custom scorer
    rmse = rmse_scorer(y_test, y_test_pred)
    
    # Append the model name and RMSE to the DataFrame
    df = pd.concat([df, pd.DataFrame({'Model': [model_name], 'RMSE': [rmse]})], ignore_index=True)
    
    return df

# Define function for calculating adjusted r-squared
def adjusted_r_squared(r2, X):
    adjusted_r2 = 1 - ((1 - r2) * (len(X) - 1) / (len(X) - X.shape[1] - 1))
    return adjusted_r2 

def rmse_scorer(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def error_analysis(y_test, y_pred_test):
    """Generated true vs. predicted values and residual scatter plot for models
    Args:
        y_test (array): true values for y_test
        y_pred_test (array): predicted values of model for y_test
    """
    # Calculate residuals
    residuals = y_test - y_pred_test
    # Plot real vs. predicted values
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    plt.subplots_adjust(right=1)
    plt.suptitle('Error Analysis')
    ax[0].scatter(y_pred_test, y_test, color="#FF5A36", alpha=0.7)
    ax[0].plot([-400, 350], [-400, 350], color="#193251")
    ax[0].set_title("True vs. Predicted Values", fontsize=16)
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("True Values")
    ax[0].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[0].set_ylim((y_test.min()-40), (y_test.max()+40))
    ax[1].scatter(y_pred_test, residuals, color="#FF5A36", alpha=0.7)
    ax[1].plot([-400, 350], [0,0], color="#193251")
    ax[1].set_title("Residual Scatter Plot", fontsize=16)
    ax[1].set_xlabel("Predicted Values")
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[1].set_ylim((residuals.min()-10), (residuals.max()+10));

