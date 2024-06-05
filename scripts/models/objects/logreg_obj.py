from sklearn.linear_model import LogisticRegression
import sys
import pandas as pd
import numpy as np
import optuna
import copy
class logreg_model():
    def __init__(
            self,
            hyperparameters = None,
            penalty='elasticnet',
            solver='saga',
            multi_class='multinomial',
            random_seed=42,
            verbose=True,
            use_gpu=False,
            **kwargs):
        
        hp_l1_ratio = 0.5
        hp_max_iter = 500

        if hyperparameters is not None:
            if 'l1_ratio' in hyperparameters:
                hp_l1_ratio = float(hyperparameters['l1_ratio'])
                sys.stdout.write("Using l1_ratio: " + str(hp_l1_ratio) + "\n")
            if 'iterations' in hyperparameters:
                hp_max_iter = int(hyperparameters['iterations'])
                sys.stdout.write("Using iterations: " + str(hp_max_iter) + "\n")

        self.model = LogisticRegression(
            l1_ratio=hp_l1_ratio,
            max_iter=hp_max_iter,
            penalty=penalty,
            solver=solver,
            multi_class=multi_class,
            random_state=random_seed,
            verbose=verbose,
            **kwargs)
        
        self.df_importances = pd.DataFrame(columns=["date", "feature", "bottom_importance", "middle_importance", "top_importance"])

    def reset_model(self, **kwargs):
        hp_max_iter = self.get_params()["max_iter"]
        hp_l1_ratio = self.get_params()["l1_ratio"]

        penalty = self.get_params()["penalty"]
        solver = self.get_params()["solver"]
        random_seed = self.get_params()["random_state"]
        multi_class = self.get_params()["multi_class"]
        verbose = self.get_params()["verbose"]
        
        self.model = LogisticRegression(
            l1_ratio=hp_l1_ratio,
            max_iter=hp_max_iter,
            penalty=penalty,
            solver=solver,
            multi_class=multi_class,
            random_state=random_seed,
            verbose=verbose,
            **kwargs)
        
        self.df_importances = pd.DataFrame(columns=["date", "feature", "bottom_importance", "middle_importance", "top_importance"])       

    def copy_model(self, **kwargs):
        return copy.deepcopy(self)
        
    def fit(self, X_train, y_train, **kwargs):
        X_train = pd.get_dummies(X_train)
        self.fit_features = X_train.columns
        self.model.fit(X_train, y_train)

    def optuna_search(self, X, y, cv_indicies, n_trials = 50, param_ranges=None, par_jobs=1, **kwargs):
        def objective(trial):

            if param_ranges is None:
                params = {
                    'max_iter' : trial.suggest_int('max_iter', 1000, 1500),
                    'l1_ratio' : trial.suggest_float('l1_ratio', 0.1, 0.9)
                }

            else:
                params = {
                    'max_iter' : trial.suggest_int('max_iter', param_ranges["iterations"][0], param_ranges["iterations"][1]),
                    'l1_ratio' : trial.suggest_float('l1_ratio', param_ranges["l1_ratio"][0], param_ranges["l1_ratio"][1])
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

                current_model.fit(X_train, y_train)

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
        return self.model.predict_proba(X)
    
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
    
    def save_model(self, path, **kwargs):
        import pickle
        with open(path + "svm_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
    