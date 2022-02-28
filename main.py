import pathlib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
from time import sleep
import pandas as pd


ROOT = pathlib.Path().resolve()
DATASETS = ROOT / "Input-Dataset"
MODELS = ROOT / "Models"



def latest_file(path=MODELS, model_type="RandomForest"):
    files = [x for x in path.glob("*.pickle") if model_type in str(x)]
    if not files:
        raise FileNotFoundError("\nMODEL DOES NOT EXIST FOR THIS CLASSIFIER YET!\nTrain and save a model using the training.py file")
    return max(files, key=lambda x: x.stat().st_ctime)

class Satisfaction:
    def __init__(self):
        self.load_model()
        
    def load_model(self, model_name=None):
        """
        If not model_name is provided, the latest model will be loaded.

        Args:
            model_name ([type], optional): [description]. Defaults to None.
        """
        model_type = input("\nWELCOME!\n\nWhat classifier do you want to load? (StackClassifier, ETEnsemble, RandomForest, KNN, NeuralNetwork or SVM) Default='RandomForest': ")
        print(f"Loading latest {model_type} model...")
        
        if not model_name:
            model_name = pathlib.Path(str(latest_file(model_type=model_type)) if model_type else str(latest_file()))
        
        with open(model_name, "rb") as f:
            self.trained_model = pickle.load(f)  
        print(f"""
==========================================
MODEL TYPE: {model_type}
MODEL FILENAME: {model_name.name}
Model Accuracy is: {round(self.trained_model.test_score * 100, 4)}%
=========================================""")
        

            
    def predict(self, X:list, scaled:bool=False):
        # if len(X)!= self.trained_model.n_features_in_:
        #     X = None
        #     print(Warning("\nThe number of features don't match, enter features below\n\n"))
        #     sleep(2)
        if not X:
            print("\n******************************************************\n")
            print("You will need to retrieve feature names for prediction in the following order\n\n", "\n".join(self.trained_model.feature_names_in_))
            X = input("\nEnter List, Dictionary or DataFrame of features below (set scaled=True if the input data is scaled): ")
        
        if type(X) == list:
            if type(X[0]) == dict:
                inp_X = pd.DataFrame.from_dict(X)
            else:
                inp_X = np.asarray(X).reshape(1, -1)

        elif type(X) == dict:
            inp_X = pd.DataFrame.from_dict(X[0], orient='index').T
            
        out = ["Unsatisfied or Neutral", "Satisfied"]
        if not scaled:
            inp_X = StandardScaler().fit_transform(inp_X)
        x = inp_X.values if type(inp_X)==pd.core.frame.DataFrame else inp_X
        res = self.trained_model.predict(x)[0]
        res_df = pd.DataFrame(columns=["Correct Class".ljust(25), "Predicted Result".ljust(25)])
        res_df.loc[0] = ["Unsatisfied or Neutral".ljust(25), str(out[int(res)]).ljust(25)]

        print(f"TEST PREDICTION\n\n{res_df.to_string(index=False)}")
        return out[int(res)]
    
    def preprocess(self, row):
        
        encodings = self.df.groupby('Class')['Satisfaction'].mean().reset_index().rename(columns={"Satisfaction":"Encoded_Class"})
        self.target_encoded = self.df.merge(encodings, how='left', on='Class')
        self.target_encoded.drop('Class', axis=1, inplace=True)
        
    
if __name__ == "__main__":
    sat = Satisfaction()
    print("\n******************************************************\n")
    sat.predict(X=[{'Customer Type': 1.0,
 'Age': 0.2692307692307692,
 'Type of Travel': 0.0,
 'Wifi Service': 0.75,
 'Online Booking': 0.75,
 'Online Boarding': 0.75,
 'Seat Comfort': 1.0,
 'On-board Service': 0.75,
 'Leg Room': 1.0,
 'Baggage': 0.5,
 'Checkin Service': 0.0,
 'Inflight Service': 0.5,
 'Cleanliness': 1.0,
 'Departure Delay': 0.0026595744680851063,
 'Arrival Delay': 0.007174887892376682
 }], scaled=True)
    print("\n******************************************************\n")
    

    # print(sat.trained_model.score(self.x_test, self.y_test))
    
