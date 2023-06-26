import pandas as pd,os,re
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    PositiveReals,
    Objective,
    Constraint,
    maximize,
    SolverFactory,
)

class Category(Enum):
    CLF=1,
    REG=2

    @staticmethod
    def get_named_category(category):
        return category.name

    @staticmethod
    def get_category(named_category):
        if named_category.lower()=='classification':
            return Category.CLF
        elif named_category.lower()=='regression':
            return Category.REG
        return None

class Dataset:
    configurations=None
    path_to_datasets=os.path.join('..','datasets')

    @staticmethod
    def load_configurations():
        Dataset.configurations=pd.read_csv(os.path.join('.','..','datasets_db.csv'))

    def __init__(self):
        self.identifier=None
        self.patterns=list()
        self.data=None
        self.category=None
    
    def read(self,filename):
        extension=filename[filename.index("."):]
        self.identifier=filename.replace(extension,"")
        self.category=Category.get_category(Dataset.configurations[Dataset.configurations['Instance']==self.identifier]['Category'])
        separator=Dataset.configurations[Dataset.configurations['Instance']==self.identifier]['Separator']
        has_categorical_data=True if Dataset.configurations[Dataset.configurations['Instance']==self.identifier]['Categorical']=='True' else False
        self.data=pd.read_csv(Dataset.path_to_datasets,delimiter=separator)
        if has_categorical_data:
            label_encoded_data=self.data[self.data.columns.to_list()[-1]]
            encoder=LabelEncoder()
            label_encoded_data=encoder.fit_transform(label_encoded_data)
            self.data[self.data.columns.to_list()[-1]]=label_encoded_data
    
    def no_patterns(self):
        if self.category==Category.CLF:
            return len(list(set(self.data.columns.to_list()[-1])))
        return -1
