import pandas as pd,os
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self,filename,seperator):
        self.data=pd.read_csv(filepath_or_buffer=filename,delimiter=seperator)
        self.filename=filename

    def convert(self):
        print('Columns[')
        for column_name in self.data.columns.to_list():
            contains_strings=pd.to_numeric(self.data[column_name],errors='coerce').isna().any() 
            if contains_strings:
                extracted_data=self.data[column_name].to_list()
                encoder=LabelEncoder()
                extracted_data=encoder.fit_transform(extracted_data)
                self.data[column_name]=extracted_data
                print(f'{column_name}',end=' ')
        print(']')
    
    def save(self):
        self.data.to_csv(self.filename,index=None)

if __name__=='__main__':
    dataset=Dataset(os.path.join('','datasets','forestfires.csv'),seperator=',')
    dataset.convert()
    dataset.save()