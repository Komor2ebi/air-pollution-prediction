# import sys
# # setting path
# sys.path.append('../')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .evaluation import print_scores

# from ..evaluation.evaluation_functions import print_scores

# def train_grid_cv(X_train,X_test,y_train,y_test,regressor,preprocessor,search_params,search_type='grid',reg_name='no_name',scoring_func = 'r2'):
def train_grid_cv(X_train,X_test,y_train,y_test,pipe_model,search_params,search_type='grid',reg_name='no_name',scoring_func = 'r2'):  
    
    # Building a full pipeline with our preprocessor and ridge regressor
    
    # pipe_model = Pipeline([
    # ('preprocessor', preprocessor),
    # (reg_name, regressor)
    # ])
    
    if search_type == 'grid':
        estimator = GridSearchCV(
            pipe_model, param_grid=search_params, cv=5,
            scoring=scoring_func, verbose=0, n_jobs=-1)
    else:
        estimator = RandomizedSearchCV(
            pipe_model, param_distributions=search_params, cv=5,
            scoring=scoring_func, verbose=0, n_jobs=-1)
        
    estimator.fit(X_train, y_train)
    
    # Show best parameters
    print('Best score:\n{:.2f}'.format(estimator.best_score_))
    print("Best parameters:\n{}".format(estimator.best_params_))
    
    # Save best model (including fitted preprocessing steps) as best_model
    best_estimator = estimator.best_estimator_
    
    # Predict using the optimized model
    y_pred = best_estimator.predict(X_test)
    
    # Calculating Error measures
    print_scores(X_test, y_test, y_pred)  

    return best_estimator, y_pred



