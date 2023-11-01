from sklearn.model_selection import ParameterGrid




class ModelWorks():
    """Class to handle hyperparameter tuning given input ModelSpec classes"""

    def __init__(self, model_specs, data):
        self.model_specs = {spec.name:spec for spec in model_specs}
        self.data = data
    

    def preprocess(self, spec):
        data = self.data
        for transform in spec.preprocessing.values:
            data = transform(data)
        return data
    

    @staticmethod
    def grid_search_trials(spec):