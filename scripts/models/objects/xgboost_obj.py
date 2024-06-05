import xgboost as xgb
import sys
import pandas as pd
import numpy as np
import optuna
import copy

class xgb_class_model():
    def __init__(self,
                 hyperparameters = None,
                 objective='multi:softmax',
                 eval_metric='mlogloss',
                 random_seed=42,
                 tree_method='hist',
                 use_gpu=False,
                 verbose=True,
                 **kwargs):
        
        hp_iterations = 1000
        hp_learning_rate = 0.1
        hp_depth = 8
        
        if hyperparameters is not None:
            if 'iterations' in hyperparameters:
                hp_iterations = int(hyperparameters['iterations'])
                sys.stdout.write("Using iterations: " + str(hp_iterations) + "\n")
            if 'learning_rate' in hyperparameters:
                hp_learning_rate = float(hyperparameters['learning_rate'])
                sys.stdout.write("Using learning_rate: " + str(hp_learning_rate) + "\n")
            if 'depth' in hyperparameters:
                hp_depth = int(hyperparameters['depth'])
                sys.stdout.write("Using depth: " + str(hp_depth) + "\n")

        self.model = xgb.XGBClassifier(
            n_estimators=hp_iterations,
            learning_rate=hp_learning_rate,
            max_depth=hp_depth,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_seed,
            tree_method=tree_method,
            device="cuda" if use_gpu else "cpu",
            verbosity= 2 if verbose else 0,
            enable_categorical=True,
            **kwargs)
        
        self.df_importances = pd.DataFrame(columns=["date", "feature", "weight", "gain", "cover", "total_gain", "total_cover"])

    def reset_model(self, **kwargs):
        hp_iterations = self.get_params()["n_estimators"]
        hp_learning_rate = self.get_params()["learning_rate"]
        hp_depth = self.get_params()["max_depth"]
        objective = self.get_params()["objective"]
        tree_method = self.get_params()["tree_method"]
        eval_metric = self.get_params()["eval_metric"]
        random_seed = self.get_params()["random_state"]
        device = self.get_params()["device"]
        verbosity = self.get_params()["verbosity"]
        
        self.model = xgb.XGBClassifier(
            n_estimators=hp_iterations,
            learning_rate=hp_learning_rate,
            max_depth=hp_depth,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_seed,
            tree_method=tree_method,
            device=device,
            verbosity=verbosity,
            enable_categorical=True,
            **kwargs)

        self.df_importances = pd.DataFrame(columns=["date", "feature", "weight", "gain", "cover", "total_gain", "total_cover"])

    def copy_model(self, **kwargs):
        return copy.deepcopy(self)

    def fit(self, X_train, y_train, X_val = None, y_val = None, cat_features = None, **kwargs):
        y_train = y_train - y_train.min() #From -1, 0, 1 to 0, 1, 2
        if X_val is not None:
            y_val = y_val - y_val.min() #From -1, 0, 1 to 0, 1, 2
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            self.model.fit(X_train, y_train)

    def optuna_search(self, X, y, cv_indicies, n_trials = 50, param_ranges=None, par_jobs=1, **kwargs):
        def objective(trial):

            if param_ranges is None:
                params = {
                    'n_estimators' : trial.suggest_int('n_estimators', 1000, 1500),
                    'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.05),
                    'max_depth' : trial.suggest_int('max_depth', 6, 8),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 0.0, 10),
                }

            else:
                params = {
                    'n_estimators' : trial.suggest_int('n_estimators', param_ranges["iterations"][0], param_ranges["iterations"][1]),
                    'learning_rate' : trial.suggest_float('learning_rate', param_ranges["learning_rate"][0], param_ranges["learning_rate"][1]),
                    'max_depth' : trial.suggest_int('max_depth', param_ranges["depth"][0], param_ranges["depth"][1]),
                    'reg_lambda' : trial.suggest_float('reg_lambda', param_ranges["l2"][0], param_ranges["l2"][1]),
                }

            sys.stdout.write(f"Trying params: {params}\n")
            current_model = self.copy_model()

            current_model.reset_model()
            current_model.model.set_params(**params)
            current_model.model.set_params(verbosity=0)

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
        self.model.set_params(verbosity=2)

        return best_params, study.trials_dataframe()
    
    def predict(self, X, **kwargs):
        preds = self.model.predict(X)
        preds = preds - len(np.unique(preds))//2 #From 0, 1, 2 to -1, 0, 1
        return preds

    def predict_proba(self, X, **kwargs):
        preds_proba = self.model.predict_proba(X)
        return preds_proba
    
    def set_new_feature_importance(self, date, features, **kwargs):

        weight_importance = self.model.get_booster().get_score(importance_type="weight")
        gain_importance = self.model.get_booster().get_score(importance_type="gain")
        cover_importance = self.model.get_booster().get_score(importance_type="cover")
        total_gain_importance = self.model.get_booster().get_score(importance_type="total_gain")
        total_cover_importance = self.model.get_booster().get_score(importance_type="total_cover")

        df_importance = pd.DataFrame(columns=["date", "feature", "weight", "gain", "cover", "total_gain", "total_cover"])

        df_importance["feature"] = features
        df_importance["date"] = date
        df_importance["weight"] = df_importance["feature"].map(weight_importance)
        df_importance["gain"] = df_importance["feature"].map(gain_importance)
        df_importance["cover"] = df_importance["feature"].map(cover_importance)
        df_importance["total_gain"] = df_importance["feature"].map(total_gain_importance)
        df_importance["total_cover"] = df_importance["feature"].map(total_cover_importance)

        self.df_importances = pd.concat([self.df_importances, df_importance])
    
    def get_feature_importances(self, **kwargs):
        return self.df_importances
    
    def get_mean_feature_importance(self, **kwargs):
        return self.df_importances.groupby(["feature"])[["weight", "gain", "cover", "total_gain", "total_cover"]].mean().sort_values(by="gain", ascending=False)
    
    def get_params(self, **kwargs):
        return self.model.get_params()
    
    def save_model(self, path, **kwargs):
        self.model.save_model(path + "xgboost_model.json")
