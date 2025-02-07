import pandas as pd
import numpy as np
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from data_transformation import DataTransformation
import pickle
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import logging

try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        RandomForestClassifier,
    )
    from sklearn.svm import SVR, SVC
    from sklearn.preprocessing import StandardScaler

except ImportError as err: 
    logging.ERROR("Module not found " + str(err))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, problem_type):#: str = "Regression"
        """
        Initialize ModelTrainer
        
        Args:
            problem_type: "Regression" or "Classification"
        """
        self.model_config = ModelTrainerConfig()
        self.problem_type = problem_type
        self.models = self._get_models()
        self.param_grids = self._get_param_grids()
        
    def _get_models(self) -> Dict:
        """
        Define models based on problem type
        """
        if self.problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "SVR": SVR(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),               
            }
        else:  # classification
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVC": SVC(random_state=42),
            }
        return models
    
    def _get_param_grids(self) -> Dict:
        """
        Define parameter grids for GridSearchCV
        """
        if self.problem_type == "Regression":
            param_grids = {
                "Linear Regression": {},
                "Decision Tree": {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                },
                "SVR": {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                },
                "XGBoost": {
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            }
        else:  # classification
            param_grids = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10]
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                },
                "XGBoost": {
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            }
        return param_grids

    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate all models and return their performances
        """
        model_performances = {}
        
        for model_name, model in self.models.items():
            try:
                logging.info(f"Training {model_name}...")
                
                # Perform GridSearchCV if parameters exist
                if self.param_grids[model_name]:
                    grid_search = GridSearchCV(
                        model,
                        self.param_grids[model_name], 
                        cv=5,
                        scoring='r2' if self.problem_type == "Regression" else 'accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    best_model = model
                    best_params = {}
                    best_model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                # Calculate metrics
                if self.problem_type == "Regression":
                    train_score = r2_score(y_train, y_train_pred)
                    test_score = r2_score(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    
                    model_performances[model_name] = {
                        "best_params": best_params,
                        "train_r2": train_score,
                        "test_r2": test_score,
                        "test_rmse": test_rmse,
                        "model": best_model
                    }
                else:
                    train_score = accuracy_score(y_train, y_train_pred)
                    test_score = accuracy_score(y_test, y_test_pred)
                    class_report = classification_report(y_test, y_test_pred)
                    
                    model_performances[model_name] = {
                        "best_params": best_params,
                        "train_accuracy": train_score,
                        "test_accuracy": test_score,
                        "classification_report": class_report,
                        "model": best_model
                    }
                
                logging.info(f"{model_name} training completed. Test score: {test_score:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
                
        return model_performances
    
    def get_best_model(self, model_performances: Dict) -> Tuple[str, Any]:
        """
        Get the best performing model based on test score
        """
        best_score = -np.inf
        best_model_name = None
        
        for model_name, performance in model_performances.items():
            score = (performance['test_r2'] if self.problem_type == "Regression" 
                    else performance['test_accuracy'])
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, model_performances[best_model_name]['model']
    
    def save_model(self, model: Any):
        """
        Save the trained model
        """
        try:
            os.makedirs(os.path.dirname(self.model_config.trained_model_file_path), exist_ok=True)
            
            with open(self.model_config.trained_model_file_path, 'wb') as f:
                pickle.dump(model, f)
            
            logging.info("Model saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            # raise Exception(e,sys)
    
    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict:
        """
        Main method to initiate model training
        """
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )
            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info("Evaluating models")
            model_performances = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            # Get best model
            best_model_name, best_model = self.get_best_model(model_performances)
            logging.info(f"Best performing model: {best_model_name}")
            
            # Save the model
            self.save_model(best_model)
            print("best_model : ", best_model_name)
            return model_performances
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")