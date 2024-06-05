import catboost
import sys
import pandas as pd
import numpy as np
import optuna
import copy
import copy
class catboost_class_model():
    def __init__(self,
                 hyperparameters=None,
                 loss_function='MultiClass',
                 eval_metric='MultiClass',
                 random_seed=42,
                 use_gpu=False,
                 verbose=True,
                 **kwargs):
        hp_iterations = 1000
        hp_learning_rate = 0.1
        hp_depth = 6
        hp_l2_leaf_reg = None
        
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
            if 'l2_leaf_reg' in hyperparameters:
                hp_l2_leaf_reg = float(hyperparameters['l2_leaf_reg'])
                sys.stdout.write("Using l2_leaf_reg: " + str(hp_l2_leaf_reg) + "\n")

        self.model = catboost.CatBoostClassifier(
            iterations=hp_iterations,
            learning_rate=hp_learning_rate,
            depth=hp_depth,
            l2_leaf_reg=hp_l2_leaf_reg,
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=random_seed,
            task_type="GPU" if use_gpu else "CPU",
            boosting_type="Plain" if use_gpu else "Ordered",
            verbose=verbose,
            **kwargs)

        self.df_importances = pd.DataFrame(columns=["date", "feature", "importance"])
        
    def reset_model(self, **kwargs):
        hp_iterations = self.model.get_params()["iterations"]
        hp_learning_rate = self.model.get_params()["learning_rate"]
        hp_depth = self.model.get_params()["depth"]
        task_type = self.model.get_params()["task_type"]

        hp_l2_leaf_reg = None
        if "l2_leaf_reg" in self.model.get_params():
            hp_l2_leaf_reg = self.model.get_params()["l2_leaf_reg"]
            
        loss_function = self.model.get_params()["loss_function"]
        eval_metric = self.model.get_params()["eval_metric"]
        random_seed = self.model.get_params()["random_seed"]
        use_gpu = True if task_type == "GPU" else False
        verbose = self.model.get_params()["verbose"]
        
        self.model = catboost.CatBoostClassifier(
            iterations=hp_iterations,
            learning_rate=hp_learning_rate,
            depth=hp_depth,
            l2_leaf_reg=hp_l2_leaf_reg,
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=random_seed,
            task_type="GPU" if use_gpu else "CPU",
            boosting_type="Plain" if use_gpu else "Ordered",
            verbose=verbose,
            **kwargs)

        self.df_importances = pd.DataFrame(columns=["date", "feature", "importance"])

    def copy_model(self, **kwargs):
        return copy.deepcopy(self)
        
    def fit(self, X_train, y_train, X_val = None, y_val = None, plot=False, cat_features=None, **kwargs):
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, plot=plot)
        else:
            self.model.fit(X_train, y_train, cat_features=cat_features, plot=plot)
    
    def optuna_search(self, X, y, cv_indicies, n_trials = 50, param_ranges=None, cat_features=None, par_jobs=1, **kwargs):
        def objective(trial):

            if param_ranges is None:
                params = {
                    'iterations' : trial.suggest_int('iterations', 1000, 1500),
                    'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.05),
                    'depth' : trial.suggest_int('depth', 6, 8),
                    'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 1, 10)
                }

            else:
                params = {
                    'iterations' : trial.suggest_int('iterations', param_ranges["iterations"][0], param_ranges["iterations"][1]),
                    'learning_rate' : trial.suggest_float('learning_rate', param_ranges["learning_rate"][0], param_ranges["learning_rate"][1]),
                    'depth' : trial.suggest_int('depth', param_ranges["depth"][0], param_ranges["depth"][1]),
                    'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', param_ranges["l2"][0], param_ranges["l2"][1])
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
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)
    
    def set_new_feature_importance(self, date, features=None, **kwargs):
        importance = self.model.get_feature_importance(prettified=True)
        df_importance = pd.DataFrame({"date": date, "feature": importance["Feature Id"], "importance": importance["Importances"]})
        self.df_importances = pd.concat([self.df_importances, df_importance])
    
    def get_feature_importances(self, **kwargs):
        return self.df_importances
    
    def get_mean_feature_importance(self, **kwargs):
        return self.df_importances.groupby(["feature"])[["importance"]].mean().sort_values(by="importance", ascending=False)
    
    def get_params(self, **kwargs):
        return self.model.get_all_params()
    
    def get_model(self, **kwargs):
        return self.model
    
    def save_model(self, path, **kwargs):
        self.model.save_model(path + "catboost_model.cbm", format="cbm")
        self.model.save_model(path + "catboost_model.json", format="json")
        