import pandas as pd

class Unimplementable:

    def unimplemented(self, func_name):
        print(str(type(self)) + ': ' + func_name + ' is not yet implemented!')
        raise

class Initable(Unimplementable):

    def initialize(self):
       self.unimplemented(self.initialize.__name__)

class DependencyInjector:

    def model_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def inject(self, file_name):
        models_description = pd.read_csv(file_name, encoding='utf-8')
        models = {}
        for index, row in models_description.iterrows():
            temp_model_name = row['model_name']
            if temp_model_name == 'not_installed':
                continue
            model_class = self.model_import(temp_model_name)
            models[row['module_name']] = model_class()
        return models

class Facade(Unimplementable):

    def __init__(self):
        self.injector = DependencyInjector()
        self.models = self.injector.inject('models.csv')
        for key, value in self.models.items():
            value.initialize()

    def get_probability(self, high, low, second, minute, hour, week_day, week_number, month, time):
        return self.models['probability_module'].get_probability(high, low, second, minute, hour, week_day, week_number, month, time)

    def get_cost(self, time):
        return self.models['cost_module'].get_cost(time)
