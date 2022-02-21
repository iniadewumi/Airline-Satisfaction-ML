import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
        self.cols = [
                       # 'Gender',
                      'Customer Type',
                      'Age',
                      'Type of Travel',
                      # 'Flight Distance',
                      'Wifi Service',
                      # 'Departure/Arrival',
                      'Online Booking',
                      # 'Gate Location',
                       # 'Food and Drink',
                       'Online Boarding',
                      'Seat Comfort',
                      # 'Inflight Entertainment',
                      'On-board Service',
                       'Leg Room',
                      'Baggage',
                     'Checkin Service',
                     'Inflight Service',
                      'Cleanliness',
                      'Departure Delay',
                     'Arrival Delay',
                     'Satisfaction',
                     'Encoded_Class'
                     ]
        self.inp = self.inp[self.cols]
        self.rescale_and_split()

        
#['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance', 'Wifi Service', 'Departure/Arrival', 'Online Booking', 'Gate Location', 'Food and Drink', 'Online Boarding', 'Seat Comfort', 'Inflight Entertainment', 'On-board Service', 'Leg Room', 'Baggage', 'Checkin Service', 'Inflight Service', 'Cleanliness', 'Departure Delay', 'Arrival Delay', 'Satisfaction', 'Encoded_Class']

    
    def rescale_and_split(self):    
        scaled_inp = pd.DataFrame(columns=self.inp.columns)
        unscaled_X = self.inp.drop(['Satisfaction'], axis=1)
        scaled_inp[unscaled_X.columns] = pd.DataFrame(MinMaxScaler().fit_transform(unscaled_X))
        scaled_inp["Satisfaction"] = self.inp["Satisfaction"]

        self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(scaled_inp[unscaled_X.columns], scaled_inp["Satisfaction"] , test_size=0.4, random_state =21)
        self.train_classif()
        
    def train_classif(self, classifier_type=None):
        classifiers = {"RandomForest": RandomForestClassifier(), "KNN": KNeighborsClassifier(n_neighbors=11), "SVM": svm.SVC()}
        if not classifier_type:
            classifier_type=input("What classifier would you like to use for training? (RandomForest, KNN, SVM or NeuralNetwork): ")
        
        
        self.trained_model = self.init_nn() if classifier_type == "NeuralNetwork" else classifiers[classifier_type]
        print(f"Training model with {classifier_type}s...")        
        self.trained_model.fit(self.x_train, self.y_train)
        self.pred = self.trained_model.predict(self.x_test)
        print(classification_report(self.y_test, self.pred))
        print(self.trained_model.score(self.x_test, self.y_test))
        print(self.pred[:5])
        # save_or_not = ""
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

# df = pd.read_csv(DATASETS/"Cleaned-Satisfaction ().csv")
# all_cols =['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance', 'Wifi Service', 'Departure/Arrival', 'Online Booking', 'Gate Location', 'Food and Drink', 'Online Boarding', 'Seat Comfort', 'Inflight Entertainment', 'On-board Service', 'Leg Room', 'Baggage', 'Checkin Service', 'Inflight Service', 'Cleanliness', 'Departure Delay', 'Arrival Delay', 'Satisfaction', 'Encoded_Class']

# #['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance', 'Wifi Service', 'Departure/Arrival', 'Online Booking', 'Gate Location', 'Food and Drink', 'Online Boarding', 'Seat Comfort', 'Inflight Entertainment', 'On-board Service', 'Leg Room', 'Baggage', 'Checkin Service', 'Inflight Service', 'Cleanliness', 'Departure Delay', 'Arrival Delay', 'Satisfaction', 'Encoded_Class']



# acc_df = pd.DataFrame(columns=["Col", "Acc"])
# for col in all_cols:
#     if not col:
#         continue
    
#      cols = [
#          'Gender',
#          'Customer Type',
#          'Age',
#          'Type of Travel',
#           'Flight Distance',
#          'Wifi Service',
#          'Departure/Arrival',
#          'Online Booking',
#          'Gate Location',
#          'Food and Drink',
#          'Online Boarding',
#          'Seat Comfort',
#          'Inflight Entertainment',
#          'On-board Service',
#          'Leg Room',
#          'Baggage',
#          'Checkin Service',
#          'Inflight Service',
#          'Cleanliness',
#          'Departure Delay',
#          'Arrival Delay',
#          'Satisfaction',
#          'Encoded_Class']
     
#     # ex = ["Customer Type", "Flight Distance", "Gender", "Checkin Service"]
#     inp = df[cols]
#     inp["GFD"]  = inp['Encoded_Class'] * inp["Flight Distance"]

    
#     inp = inp[[x for x in inp.columns if x not in ex]]

#     scaled_inp = pd.DataFrame(columns=inp.columns)
#     unscaled_X = inp.drop(['Satisfaction'], axis=1)
#     scaled_inp[unscaled_X.columns] = pd.DataFrame(MinMaxScaler().fit_transform(unscaled_X))
#     scaled_inp["Satisfaction"] = inp["Satisfaction"]
    
#     x_train, x_test, y_train, y_test =  train_test_split(scaled_inp[unscaled_X.columns], scaled_inp["Satisfaction"] , test_size=0.4, random_state = 42)
    
#     model = LogisticRegression()
#     model.fit(x_train, y_train)
    
#     score = model.score(x_test, y_test)
#     print(score)
#     acc_df = acc_df.append({"Col":col, "Acc":score}, ignore_index=True)
# acc_df = acc_df.sort_values("Acc")


# ['Type of Travel', 'Customer Type', 'Wifi Service', 'Online Boarding', 'Checkin Service', 'Leg Room', 'Inflight Service', 'On-board Service', 'Departure/Arrival', 'Gate Location', 'Cleanliness', 'Departure Delay', 'Baggage', 'Online Booking', 'Arrival Delay', 'Inflight Entertainment', 'Food and Drink', 'Flight Distance', 'Age', 'Seat Comfort', 'Gender']



# import statsmodels.api as sm

# #log_clf = LogisticRegression()

# log_clf =sm.Logit(self.y_train,self.x_train)

# classifier = log_clf.fit()

# y_pred = classifier.predict(self.x_test)

# print(classifier.summary())



