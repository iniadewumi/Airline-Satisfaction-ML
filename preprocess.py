import pathlib
import pandas as pd

ROOT = pathlib.Path().resolve()
DATASETS = ROOT / "Input-Dataset"
MODELS = ROOT / "Models"
COLS = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Wifi Service', 'Departure/Arrival',
       'Online Booking', 'Gate Location', 'Food and Drink', 'Online Boarding',
       'Seat Comfort', 'Inflight Entertainment', 'On-board Service',
       'Leg Room', 'Baggage', 'Checkin Service', 'Inflight Service',
       'Cleanliness', 'Departure Delay', 'Arrival Delay', 'Satisfaction']

class Cleaner:
    def __init__(self):
        self.df= pd.read_csv(DATASETS/"Raw-Satisfaction.csv")[COLS]

    def make_dummy(self):
        self.df["Arrival Delay"] = self.df["Arrival Delay"].fillna(self.df["Arrival Delay"].mean())
        self.df['Satisfaction'] = self.df['Satisfaction'].apply(lambda x: float(1) if x=="satisfied" else float(0))
        self.df["Customer Type"] = self.df["Customer Type"].apply(lambda x: float(1) if x=="Loyal Customer" else float(0))
        self.df["Gender"] = self.df["Gender"].apply(lambda x: float(1) if x=="Female" else float(0))
        self.df['Type of Travel'] =  self.df['Type of Travel'].apply(lambda x: float(1) if x=="Business travel" else float(0))
        
    def target_encode(self):
        #USING TARGET ENCODING TO PREVENT DIMENSIONALITY ISSUES
        encodings = self.df.groupby('Class')['Satisfaction'].mean().reset_index().rename(columns={"Satisfaction":"Encoded_Class"})
        self.target_encoded = self.df.merge(encodings, how='left', on='Class')
        self.target_encoded.drop('Class', axis=1, inplace=True)
        
    def one_hot_encode(self):
        self.one_hot_encoded = pd.get_dummies(self.df["Class"], drop_first=True)
        self.one_hot_encoded[self.df.loc[:, self.df.columns!="Class"].columns] = self.df.loc[:, self.df.columns!="Class"]
        
    def saver(self):
        self.target_encoded.to_csv(DATASETS/"Cleaned-Satisfaction (TE).csv", index=False)
        self.one_hot_encoded.to_csv(DATASETS/"Cleaned-Satisfaction (OHE).csv", index=False)
        
    def preprocess(self):
        self.make_dummy()
        self.target_encode()
        self.one_hot_encode()
        self.saver()
        

if __name__ == '__main__':
    cleaner = Cleaner()
    cleaner.preprocess()
    
    
# encoder = OneHotEncoder(drop='first', sparse=False)
# test = asarray(df["Customer Type"]).reshape(-1, 1)
# onehot = encoder.fit_transform(test)


# encodings = data.groupby('Country')['Target Variable'].mean().reset_index()
# data = data.merge(encodings, how='left', on='Country')
# data.drop('Country', axis=1, inplace=True)
# [method_name for method_name in dir(impute) if callable(getattr(impute, method_name))]