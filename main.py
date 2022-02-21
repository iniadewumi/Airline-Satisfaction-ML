import pathlib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
from time import sleep


ROOT = pathlib.Path().resolve()
DATASETS = ROOT / "Input-Dataset"
MODELS = ROOT / "Models"



def latest_file(path=MODELS, model_type="RandomForest"):
    files = [x for x in path.glob("*.pickle") if model_type in str(x)]
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
        model_type = input("What classifier do you want to load? (RandomForest, KNN, NeuralNetwork or SVM) Default='RandomForest': ")
        print(f"Loading latest {model_type} model...")
        if not model_name:
            model_name = str(latest_file(model_type=model_type)) if model_type else str(latest_file())
        
        with open(model_name, "rb") as f:
            self.trained_model = pickle.load(f)     
            
    def predict(self, X:list, scaled:bool=False):
        if len(X)!= self.trained_model.n_features_in_:
            X = None
            print(Warning("\nThe number of features don't match, enter features below\n\n"))
            sleep(2)
        if not X:
            print("\n******************************************************\n")
            print("You will need to retrieve feature names for prediction in the following order\n\n", "\n".join(self.trained_model.feature_names_in_))
            X = input("\nEnter List, Dictionary or DataFrame of features below: ")
        
        X = np.asarray(X).reshape(1, -1)
        out = ["Unsatisfied or Neutral", "Satisfied"]
        if not scaled:
            X = StandardScaler().fit_transform(X)
        res = self.trained_model.predict(X)[0]
        print(f"Result: {out[int(res)]}")
        return out[int(res)]
    
    def preprocess(self, row):
        
        encodings = self.df.groupby('Class')['Satisfaction'].mean().reset_index().rename(columns={"Satisfaction":"Encoded_Class"})
        self.target_encoded = self.df.merge(encodings, how='left', on='Class')
        self.target_encoded.drop('Class', axis=1, inplace=True)
        
    
if __name__ == "__main__":
    sat = Satisfaction()
    sat.predict({'Customer Type': 1.0,
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
 'Arrival Delay': 0.007174887892376682,
 'Encoded_Class': 1.0})
    
    print(sat.trained_model.score(self.x_test, self.y_test))
    
    
