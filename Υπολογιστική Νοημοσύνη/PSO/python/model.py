import numpy as np,pandas as pd,os
from enum import Enum

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Optimizer
from keras.utils import to_categorical
from keras import backend as K
from sklearn.datasets import make_classification
from pyswarms.single.global_best import GlobalBestPSO

class Category(Enum):
    CLF=1,
    REG=2,

class Base:
    datasets=pd.read_csv(os.path.join('..','datasets_db.csv'))
    path_to_datasets=os.path.join('..','datasets')

    @staticmethod
    def seperator(dataset_id):
        try:
            return Base.datasets[Base.datasets['Instance']==dataset_id]['Seperator']
        except ValueError:
            raise ValueError(f"Id {dataset_id} Not found")
    
    @staticmethod
    def category(dataset_id):
        try:
            return Category.CLF if Base.datasets[Base.datasets['Instance']==dataset_id]['Category']=="clf" else Category.REG
        except ValueError as err:
            raise err
    
    @staticmethod
    def has_header(dataset_id):
        try:
            return True if dataset_id in ['forestfires','RP_hardware_performance'] else False
        except ValueError as err:
            raise err

class Dataset:
    def __init__(self):
        self.data=None
        self.category=None
        self.id=None
        self.optimizer=None

    def read(self,filename):
        data=filename.split(os.path.sep)
        self.id=data[-1]
        self.data=pd.read_csv(os.path.join(Base.path_to_datasets,filename),delimiter=Base.seperator(self.id))
        self.category=Base.category(self.id)
        self.optimizer=Evaluation.binary_crossentropy if self.category==Category.CLF and self.no_classes()==2 else Evaluation.categorical_crossentropy if self.category==Category.CLF and self.no_classes()>2 else Evaluation.mean_squared_error 

    def set_data(self,new_pd_data):
        self.data=new_pd_data

    def set_id(self,dataset_id):
        self.id=dataset_id

    def set_category(self,category_value):
        self.category=category_value

    def get(self):
        features=self.data.columns.tolist()[:-1]
        X=self.data[features]
        Y=self.data[self.data.columns.to_list()[-1]]
        return X.to_numpy(),to_categorical(np.array(Y)) if self.no_classes()>2 else np.array(Y)

    def get_to_pandas(self):
        features=self.data.columns.tolist()[:-1]
        X=self.data[features]
        Y=self.data[self.data.columns.to_list()[-1]]
        return X,Y

    def dimension(self):
        return len(self.data.columns.to_list())-1
        
    def no_classes(self):
        return len(list(set(self.data[self.data.columns.to_list()[-1]].to_list())))

    def output_layer(self):
        no_class=self.no_classes()
        return 1 if self.category==Category.REG else no_class


class Evaluation:
    @staticmethod
    def categorical_crossentropy(y_true,y_pred):
        return K.mean(K.categorical_crossentropy(y_true,y_pred),axis=-1)
    
    @staticmethod
    def mean_squared_error(y_true,y_pred):
        return K.mean(K.square(y_pred-y_true),axis=-1)
    
    @staticmethod
    def binary_crossentropy(y_true,y_pred):
        return K.mean(K.binary_crossentropy(y_true,y_pred),axis=-1)

    

class MlpProblem:
    def __init__(self):
        self.xset=None
        self.yset=None

        self.model=Sequential()
        self.model.add(Dense(32,activation='sigmoid',input_dim=self.data.dimension()))
        self.model.add(Dense(self.dataset.output_layer(),activation='sigmoid'))

    def fitness(self,position):
        self.model.set_weights(position.reshape(self.model.get_weights().shape))
        self.model.compile(optimizer=Optimizer(),loss=self.dataset.optimizer)
        history=self.model.fit(
            self.xset,
            self.yset,
            epochs=50,
            verbose=True
        )
        return history.history['loss'][-1]
    
    def set_data(self,x,y):
        self.xset=x
        self.yset=y

    def train(self):
        options={'c1':2.0,'c2':2.0,'w':0.9}
        optimizer=GlobalBestPSO(n_particles=50,dimensions=self.model.count_params(),options=options)
        best_position,best_cost=optimizer.optimize(self.fitness,iters=5000)
        self.model.set_weights(best_position.reshape(self.model.get_weights().shape))
        X,Y=self.dataset.get()
        loss=self.model.evaluate(X,Y)
        print(f'Optimizer:PSO\tLoss:{loss}')
    
if __name__=='__main__':
    filename='forestfires.csv'
    dataset=Dataset()
    dataset.read(filename)

    cv_model=StratifiedKFold(n_splits=10)
    X,Y=dataset.get_to_pandas()
    for train_indeces,test_indeces in cv_model.split():