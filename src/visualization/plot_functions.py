import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_residuals(y,y_pred,reg_name,x_limits=None,y_limits=None):
    # Plotting the residuals
    residuals = y - y_pred # Calculate residuals
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    sns.regplot(x=y_pred, y=residuals,
                lowess=True, scatter=False, line_kws={'color': 'red', 'lw': 1}) # Add a lowess line using regplot
    plt.title(f'{reg_name} Regression Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    if x_limits != None:
        plt.xlim(*x_limits) # Set y-axis limits
    
    if y_limits != None:
        plt.ylim(*y_limits) # Set y-axis limits
        
    plt.show()
    
    return

def get_feature_names(column_transformer):
    """Get feature names from all transformers in a ColumnTransformer."""
    new_features = []  # List to collect new feature names
    for name, transformer, original_features in column_transformer.transformers_:
        if transformer == 'drop' or (hasattr(transformer, 'remainder') and transformer.remainder == 'drop'):
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            new_features.extend(transformer.get_feature_names_out(original_features))
        else:
            new_features.extend(original_features)  # Assume no change in feature names
    return new_features

def plot_most_important_features(model,model_name,number_of_features=20):
    # Assuming 'best_model_ridge' is the fitted best estimator from GridSearchCV
    coefs = model.named_steps[model_name].coef_
    preprocessor = model.named_steps['preprocessor']

    # Get new feature names after transformation
    feature_names = get_feature_names(preprocessor)

    # Sometimes the output is a single array in a multioutput regression
    if coefs.ndim > 1:
        coefs = coefs.flatten()

    # Create a pandas Series to view the feature importances
    feature_importances = pd.Series(data=coefs, index=feature_names)

    # Sort the feature importances by absolute value
    sorted_features = feature_importances.abs().sort_values(ascending=False)

    # Plotting the top 20 feature importances
    plt.figure(figsize=(10, 10))
    sorted_features.head(number_of_features).plot(kind='barh', title='Top 20 Feature Importances from Ridge Regression')
    plt.xlabel('Absolute Coefficient Value')
    plt.show()
    return

