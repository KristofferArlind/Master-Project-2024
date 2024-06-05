from objects.catboost_obj import catboost_class_model
from objects.logreg_obj import logreg_model
from objects.xgboost_obj import xgb_class_model
from objects.svm_obj import svm_class_model
from objects.randomforest_obj import rf_class_model

def choose_model(model_name):
    if model_name == "xgb":
        return xgb_class_model
    if model_name == "logreg":
        return logreg_model
    if model_name == "catboost":
        return catboost_class_model
    if model_name == "svm":
        return svm_class_model
    if model_name == "rf":
        return rf_class_model
    else:
        raise Exception("Model not found")
