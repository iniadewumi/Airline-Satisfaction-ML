import pathlib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np


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
            model_name = str(latest_file())
        with open(model_name, "rb") as f:
            self.trained_model = pickle.load(f)     
            
    def predict(self, X:list, scaled:bool=False):
        X = np.asarray(X).reshape(1, -1)
        out = ["Unsatisfied or Neutral", "Satisfied"]
        if not scaled:
            scaled_inp = StandardScaler().fit_transform(X)
        res = self.trained_model.predict(X)[0]
        print(f"Result: {out[int(res)]}")
        return out[int(res)]
        
    
if __name__ == "__main__":
    X = [1.0, 1.0, 52.0, 1.0, 160.0, 5.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0, 50.0, 44.0, 0.1938775510204081]
    sat = Satisfaction()
    sat.predict(X)
    
    
