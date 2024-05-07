import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def error_analysis_plot(y_test, y_pred_test, colorcode="#FF5A36"):
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
    sns.regplot(x=y_pred_test, y=residuals,lowess=True, scatter=False, line_kws={'color': 'blue', 'lw': 1}) # Add a lowess line using regplot
    plt.suptitle('Error Analysis')
    ax[0].scatter(y_pred_test, y_test, color=colorcode, alpha=0.7)
    ax[0].plot([-400, 350], [-400, 350], color="#193251")
    ax[0].set_title("True vs. predicted values", fontsize=16)
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("True Values")
    ax[0].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[0].set_ylim((y_test.min()-40), (y_test.max()+40))
    ax[1].scatter(y_pred_test, residuals, color=colorcode, alpha=0.7)
    ax[1].plot([-400, 350], [0,0], color="#193251")
    ax[1].set_title("Residual Scatter Plot", fontsize=16)
    ax[1].set_xlabel("Predicted Values")
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[1].set_ylim((residuals.min()-10), (residuals.max()+10));
    
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
    sorted_features.head(number_of_features).plot(kind='barh', title='Top 20 Feature Importances from Ridge Regression', color="#96c6da")
    plt.xlabel('Absolute Coefficient Value')
    plt.show()
    return

