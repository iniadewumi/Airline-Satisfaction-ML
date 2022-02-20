import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle

ROOT = pathlib.Path().resolve()
DATASETS = ROOT / "Input-Dataset"
MODELS = ROOT / "Models"

def latest_file(path=MODELS, model_type="RandomForest"):
    files = [x for x in path.glob("*.pickle") if model_type in str(x)]
    return max(files, key=lambda x: x.stat().st_ctime)



class Training:
    def __init__(self):
        self.inp = pd.read_csv(DATASETS/"Cleaned-Satisfaction (TE).csv")
        self.rescale_and_split()
    
    def rescale_and_split(self):    
        scaled_inp = pd.DataFrame(columns=self.inp.columns)
        unscaled_X = self.inp.drop(['Satisfaction'], axis=1)
        scaled_inp[unscaled_X.columns] = pd.DataFrame(MinMaxScaler().fit_transform(unscaled_X))
        scaled_inp["Satisfaction"] = self.inp["Satisfaction"]

        self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(scaled_inp[unscaled_X.columns], scaled_inp["Satisfaction"] , test_size=0.4, random_state = 42)
        self.train_classif()
        
    def train_classif(self):
        classifiers = {"RandomForest": RandomForestClassifier(), "KNN": KNeighborsClassifier(n_neighbors=11), "SVM": svm.SVC()}
        classifier_type=input("What classifier would you like to use for training? (RandomForest, KNN, SVM or NeuralNetwork): ")
        
        
        self.trained_model = self.init_nn() if classifier_type == "NeuralNetwork" else classifiers[classifier_type]
        print(f"Training model with {classifier_type}s...")        
        self.trained_model.fit(self.x_train, self.y_train)
        self.pred = self.trained_model.predict(self.x_test)
        print(classification_report(self.y_test, self.pred))
        print(self.trained_model.score(self.x_test, self.y_test))
        
        save_or_not = input("Save Model? (y or n) ")
        if save_or_not == "y":
            models_count = sum(x.suffix==".pickle" and classifier_type in x.name for x in MODELS.iterdir())
            model_name = MODELS/f"{classifier_type}({models_count}).pickle"
            self.save_model(self.trained_model, model_name)
            print(f"\n\nModel saved as {model_name.name}")
        else:
            print("Model not saved")
        
    
    def init_nn(self):
        layers = (100,50)
        inp = input(f"Enter layers as tuple for NeuralNetwork, leave blank for default {layers}: ")
        layers = eval(inp) if inp else layers
        if type(layers)!=tuple:
            raise Exception("Invalid Layers! Please retry")

        return MLPClassifier(activation="relu", hidden_layer_sizes=layers, verbose=True, max_iter=400)
        
        

    def save_model(self, model, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(model, f)

            
# if __name__ == '__main__':
#     train = Training()
self = Training()

