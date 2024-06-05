from sklearn import svm
import sys
import pandas as pd
import numpy as np
import optuna
import copy

class svm_class_model():
    def __init__(self,
                 hyperparameters=None,
                 random_seed=42,
                 use_gpu=False,
                 verbose=True,
                 **kwargs):
        hp_loss = "squared_hinge"
        hp_penalty = "l2"
        hp_C = 1.0
        hp_max_iter = 1000
        
        if hyperparameters is not None:
            if 'loss' in hyperparameters:
                hp_loss = hyperparameters['loss']
                sys.stdout.write("Using loss: " + str(hp_loss) + "\n")
            if 'penalty' in hyperparameters:
                hp_penalty = hyperparameters['penalty']
                sys.stdout.write("Using penalty: " + str(hp_penalty) + "\n")
            if 'C' in hyperparameters:
                hp_C = float(hyperparameters['C'])
                sys.stdout.write("Using C: " + str(hp_C) + "\n")
            if 'iterations' in hyperparameters:
                hp_max_iter = int(hyperparameters['iterations'])
                sys.stdout.write("Using iterations: " + str(hp_max_iter) + "\n")

        self.model = svm.LinearSVC(
            penalty=hp_penalty,
            loss=hp_loss,
            C=hp_C,
            max_iter=hp_max_iter,
            dual=False, #As there are more samples than features
            random_state=random_seed,
            verbose= 1 if verbose else 0,
            **kwargs
        )

        self.df_importances = pd.DataFrame(columns=["date", "feature", "bottom_importance", "middle_importance", "top_importance"])                      
        
    def reset_model(self, **kwargs):
        hp_loss = self.model.get_params()["loss"]
        hp_penalty = self.model.get_params()["penalty"]
        hp_C = self.model.get_params()["C"]
        hp_max_iter = self.model.get_params()["max_iter"]
        verbose = self.model.get_params()["verbose"]
        random_seed = self.model.get_params()["random_state"]
        
        self.model = svm.LinearSVC(
            penalty=hp_penalty,
            loss=hp_loss,
            C=hp_C,
            max_iter=hp_max_iter,
            dual=False,
            random_state=random_seed,
            verbose= 1 if verbose else 0,
            **kwargs
        )

        self.df_importances = pd.DataFrame(columns=["date", "feature", "bottom_importance", "middle_importance", "top_importance"])    

    def copy_model(self, **kwargs):
        return copy.deepcopy(self)
        
    def fit(self, X_train, y_train, plot=False, cat_features=None, **kwargs):
        X_train = pd.get_dummies(X_train)
        self.fit_features = X_train.columns
        self.model.fit(X_train, y_train)
                
    def optuna_search(self, X, y, cv_indicies, n_trials = 50, param_ranges=None, cat_features=None, par_jobs=1, **kwargs):
        def objective(trial):

            if param_ranges is None:
                params = {
                    'C' : trial.suggest_float('C', 1, 3),
                    'max_iter' : trial.suggest_int('max_iter', 1000, 2000),
                }

            else:
                params = {
                    'C' : trial.suggest_float('C', param_ranges["C"][0], param_ranges["C"][1]),
                    'max_iter' : trial.suggest_int('max_iter', param_ranges["iterations"][0], param_ranges["iterations"][1]),
                }

            sys.stdout.write(f"Trying params: {params}\n")
            current_model = self.copy_model()

            current_model.reset_model()
            current_model.model.set_params(**params)
            current_model.model.set_params(verbose=False)

            accs = []

            for i, (train_index, val_index) in enumerate(cv_indicies):
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]

                X_val = X.iloc[val_index]
                y_val = y.iloc[val_index]

                current_model.fit(X_train, y_train, cat_features=cat_features)

                predictions = current_model.predict(X_val)

                predictions.resize(1, len(predictions))

                acc = np.mean(predictions == np.array(y_val))
                accs.append(acc)

            return np.mean(accs)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(n_trials), n_jobs=int(par_jobs), gc_after_trial=True)

        self.reset_model()

        best_params = study.best_params

        sys.stdout.write(f"Best params: {best_params}\n")

        self.model.set_params(**best_params)
        self.model.set_params(verbose=True)

        return best_params, study.trials_dataframe()

    
    def predict(self, X, **kwargs):
        X = pd.get_dummies(X)
        for col in self.fit_features:
            if col not in X.columns:
                sys.stdout.write(f"Feature {col} not found in input data. Setting to 0\n")
                X[col] = 0
        for col in X.columns:
            if col not in self.fit_features:
                sys.stdout.write(f"Feature {col} not found in trained model. Dropping\n")
                X = X.drop(columns=col)
        X = X[self.fit_features]
        sys.stdout.write(f"Predicting with features: {X.columns}\n")
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        X = pd.get_dummies(X)
        for col in self.fit_features:
            if col not in X.columns:
                sys.stdout.write(f"Feature {col} not found in input data. Setting to 0\n")
                X[col] = 0
        for col in X.columns:
            if col not in self.fit_features:
                sys.stdout.write(f"Feature {col} not found in trained model. Dropping\n")
                X = X.drop(columns=col)
        X = X[self.fit_features]
        sys.stdout.write(f"Predicting with features: {X.columns}\n")
        return self.model.decision_function(X)
    
    def set_new_feature_importance(self, date, features=None, **kwargs):
        sys.stdout.write(str(self.fit_features) + "\n")
        df_importance = pd.DataFrame({"date": date, "feature": self.fit_features, "bottom_importance": self.model.coef_[0], "middle_importance": self.model.coef_[1], "top_importance": self.model.coef_[2]})
        self.df_importances = pd.concat([self.df_importances, df_importance])
    
    def get_feature_importances(self, **kwargs):
        return self.df_importances
    
    def get_mean_feature_importance(self, **kwargs):
        return self.df_importances.groupby(["feature"])[["bottom_importance", "middle_importance", "top_importance"]].mean().sort_values(by="top_importance", ascending=False)
    
    def get_params(self, **kwargs):
        return self.model.get_params()
    
    def get_model(self, **kwargs):
        return self.model
    
    def save_model(self, path, **kwargs):
        import pickle
        with open(path + "svm_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)