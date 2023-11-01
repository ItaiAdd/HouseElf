import pandas as pd




class sklearnModelSpec():
    """
    Class to store information unique to a specific (model, preprocessing)
    combination. This is specifically for models which use the Scikit-Learn.

    Attributes:
        name (str): Reference name for (model, preprocessing) combination.
        model: Model object, this must use the Scikit-Learn fit/predict
               interface.
        params: Parameter ranges for tuning. Either a full list of values for
                grid search or low/high for optuna TPE sampling.
    """

    def __init__(self,
                 name,
                 model,
                 params,
                 fit_params=None,
                 custom_fit=None,
                 custom_predict=None):
        
        self.name = name
        self.model = model
        self.params = params
        self.fit_params = fit_params
        preprocessing = None
        self.custom_fit = custom_fit
        self.custom_predict = custom_predict
        self.tuning_history = {param:[] for param in params.keys()}
        self.tuning_history['objective'] = []
        self.trained_model = None
        self.tuned_model = None


    def fit(self, par, fit_par, X, y):
        model = self.model(**par)
        model.fit(X, y, **fit_par)
        self.trained_model = model
    

    def predict(self, X):
        y_pred = self.trained_model.predict(X)
        return y_pred
    

    def predict_proba(self, X):
        y_pred = self.trained_model.predict_proba(X)
        return y_pred
    

    def update_history(self, trial_params, objective):
        self.tuning_history['objective'].append(objective)
        for param, val in trial_params.items():
            self.tuning_history[param].append(val)


    def history_df(self):
        df = pd.DataFrame(self.tuning_history)
        return df