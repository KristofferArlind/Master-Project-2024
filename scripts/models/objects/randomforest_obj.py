import sys
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
import copy

class rf_class_model():
    def __init__(self,
                 hyperparameters=None,
                 random_seed=42,
                 use_gpu=False,
                 verbose=True,
                 **kwargs):

        hp_trees = 500
        hp_max_depth = None
        hp_min_samples_split = 2
        hp_min_samples_leaf = 1
        hp_min_weight_fraction_leaf = 0.0
        hp_max_features = "sqrt"
        hp_max_samples = None
        hp_n_jobs = 1
        
        if hyperparameters is not None:
            if 'rf_jobs' in hyperparameters:
                hp_n_jobs = int(hyperparameters['rf_jobs'])
                sys.stdout.write("Using n rf jobs: " + str(hp_n_jobs) + "\n")
            if 'trees' in hyperparameters:
                hp_trees = int(hyperparameters['trees'])
                sys.stdout.write("Using trees: " + str(hp_trees) + "\n")
            if 'depth' in hyperparameters:
                hp_max_depth = int(hyperparameters['depth'])
                sys.stdout.write("Using max depth: " + str(hp_max_depth) + "\n")
            if 'min_samples_split' in hyperparameters:
                hp_min_samples_split = int(hyperparameters['min_samples_split'])
                sys.stdout.write("Using min samples split: " + str(hp_min_samples_split) + "\n")
            if 'min_samples_leaf' in hyperparameters:
                hp_min_samples_leaf = int(hyperparameters['min_samples_leaf'])
                sys.stdout.write("Using min samples leaf: " + str(hp_min_samples_leaf) + "\n")
            if 'min_weight_fraction_leaf' in hyperparameters:
                hp_min_weight_fraction_leaf = float(hyperparameters['min_weight_fraction_leaf'])
                sys.stdout.write("Using min_weight_fraction_leaf: " + str(hp_min_weight_fraction_leaf) + "\n")
            if 'max_features' in hyperparameters:
                hp_max_features = float(hyperparameters['max_features'])
                sys.stdout.write("Using max_features: " + str(hp_max_features) + "\n")
            if 'max_samples' in hyperparameters:
                hp_max_samples = float(hyperparameters['max_samples'])
                sys.stdout.write("Using max_samples: " + str(hp_max_samples) + "\n")

        self.model = RandomForestClassifier(
            n_estimators=hp_trees,
            max_depth=hp_max_depth,
            min_samples_split=hp_min_samples_split,
            min_samples_leaf=hp_min_samples_leaf,
            min_weight_fraction_leaf=hp_min_weight_fraction_leaf,
            max_features=hp_max_features,
            max_samples=hp_max_samples,
            n_jobs=hp_n_jobs,
            random_state=random_seed,
            verbose=1 if verbose else 0,
            **kwargs
        )

        self.df_importances = pd.DataFrame(columns=["date", "feature", "importance"])                      
        
    def reset_model(self, **kwargs):
        hp_trees = self.model.get_params()["n_estimators"]
        hp_max_depth = self.model.get_params()["max_depth"]
        hp_min_samples_split = self.model.get_params()["min_samples_split"]
        hp_min_samples_leaf = self.model.get_params()["min_samples_leaf"]
        hp_min_weight_fraction_leaf = self.model.get_params()["min_weight_fraction_leaf"]
        hp_max_features = self.model.get_params()["max_features"]
        hp_max_samples = self.model.get_params()["max_samples"]
        
        verbose = self.model.get_params()["verbose"]
        random_seed = self.model.get_params()["random_state"]

        self.model = RandomForestClassifier(
            n_estimators=hp_trees,
            max_depth=hp_max_depth,
            min_samples_split=hp_min_samples_split,
            min_samples_leaf=hp_min_samples_leaf,
            min_weight_fraction_leaf=hp_min_weight_fraction_leaf,
            max_features=hp_max_features,
            max_samples=hp_max_samples,
            random_state=random_seed,
            verbose=1 if verbose else 0,
            **kwargs
        ) 
        
        self.df_importances = pd.DataFrame(columns=["date", "feature", "importance"])    

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
                    'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
                    #'max_features' : trial.suggest_float('max_features', 0.1, 1.0),
                    'max_samples' : trial.suggest_float('max_samples', 0.3, 1.0),
                }

            else:
                if "max_features" in param_ranges:
                    params = {
                        'n_estimators' : trial.suggest_int('n_estimators', param_ranges["n_estimators"][0], param_ranges["n_estimators"][1]),
                        #'max_features' : trial.suggest_float('max_features', param_ranges["max_features"][0], param_ranges["max_features"][1]),
                        'max_samples' : trial.suggest_float('max_samples', param_ranges["max_samples"][0], param_ranges["max_samples"][1]),
                    }
                else:
                    params = {
                        'n_estimators' : trial.suggest_int('n_estimators', param_ranges["n_estimators"][0], param_ranges["n_estimators"][1]),
                        #'max_features' : trial.suggest_float('max_features', param_ranges["max_features"][0], param_ranges["max_features"][1]),
                        'max_samples' : trial.suggest_float('max_samples', param_ranges["max_samples"][0], param_ranges["max_samples"][1]),

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
        return self.model.predict_proba(X)
    
    def set_new_feature_importance(self, date, features=None, **kwargs):
        sys.stdout.write(str(self.fit_features) + "\n")
        df_importance = pd.DataFrame({"date": date, "feature": self.fit_features, "importance": self.model.feature_importances_})
        self.df_importances = pd.concat([self.df_importances, df_importance])
    
    def get_feature_importances(self, **kwargs):
        return self.df_importances
    
    def get_mean_feature_importance(self, **kwargs):
        return self.df_importances.groupby(["feature"])[["importance"]].mean().sort_values(by="importance", ascending=False)
   
    def get_params(self, **kwargs):
        return self.model.get_params()
    
    def get_model(self, **kwargs):
        return self.model
    
    def save_model(self, path, **kwargs):
        sys.stdout.write("Cant save model, RF becomes too big\n")
        #import pickle
        #with open(path + "svm_model.pkl", 'wb') as f:
        #    pickle.dump(self.model, f)