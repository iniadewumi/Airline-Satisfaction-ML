import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
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
        self.inp = pd.read_csv(DATASETS/"Cleaned-Satisfaction (NUM).csv")
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
                     # 'Encoded_Class'
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
    
        
    # =============================================================================
    #     ET Ensemble = ExtraTreeEnsemble Classifier
    # =============================================================================
    def train_classif(self, classifier_type=None):
        classifiers = {"ETEnsemble":ExtraTreesClassifier(random_state=0), "RandomForest": RandomForestClassifier(verbose=1), "KNN": KNeighborsClassifier(n_neighbors=7), "SVM": svm.SVC(verbose=1, C=7)}
        if not classifier_type:
            classifier_type=input("What classifier would you like to use for training? (RandomForest, ETEnsemble, KNN, SVM or NeuralNetwork): ")
        
        
        self.trained_model = self.init_nn() if classifier_type == "NeuralNetwork" else classifiers[classifier_type]
        print(f"Training model with {classifier_type}s...")        
        self.trained_model.fit(self.x_train.values, self.y_train)
        # self.pred = self.trained_model.predict(self.x_test)
        test_score = self.trained_model.score(self.x_test.values, self.y_test)
        setattr(self.trained_model, "test_score", test_score)


        # print(classification_report(self.y_test, self.pred))
        print(self.trained_model.test_score)
        # print(self.pred[:5])

        save_or_not = input("Save Model? (y or n) ")
        if save_or_not == "y":
            models_count = sum(x.suffix==".pickle" and classifier_type in x.name for x in MODELS.iterdir())
            model_name = MODELS/f"{classifier_type}({models_count}).pickle"
            self.save_model(self.trained_model, model_name)
            print(f"\n\nModel saved as {model_name.name}")
        else:
            print("Model not saved")
        
    
    def init_nn(self):
        layers = (70,30)
        inp = input(f"Enter layers as tuple for NeuralNetwork, leave blank for default {layers}: ")
        layers = eval(inp) if inp else layers
        if type(layers)!=tuple:
            raise Exception("Invalid Layers! Please retry")

        return MLPClassifier(activation="relu", hidden_layer_sizes=layers, verbose=True, max_iter=400)
    def stack_ensemble(self):
        layers = (70,30)
        estimators = [
            ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, random_state=1)),
            ('nn', MLPClassifier(activation="relu", hidden_layer_sizes=layers, verbose=True, max_iter=400)),
            ('rf', RandomForestClassifier(verbose=1)),
            ('svm', svm.SVC(verbose=1, C=7))
            ]
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        clf.fit(self.x_train.values, self.y_train)
        
        test_score = clf.score(self.x_test, self.y_test)
        setattr(clf, "test_score", test_score)
        model_name = MODELS/"StackingClassifier.pickle"
        with open(model_name, "wb") as f:
            pickle.dump(clf, f)

        

    def save_model(self, model, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(model, f)

            
# if __name__ == '__main__':
#     train = Training()
self = Training()

# ens = pd.DataFrame()

# layers = (70,30,10,10,5)
# test = MLPClassifier(activation="relu", hidden_layer_sizes=layers, alpha=0.00001, verbose=True, learning_rate_init=0.0001, max_iter=1000)
# test.fit(self.x_train.values, self.y_train)
# print(layers)
# print(test.score(self.x_test, self.y_test))

# clf.score(self.x_test, self.y_test)


# from sklearn.ensemble import BaggingClassifier, StackingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import ExtraTreesClassifier

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, random_state=1).fit(self.x_train, self.y_train)
# layers = (100,50)
# nn = MLPClassifier(activation="relu", hidden_layer_sizes=layers, verbose=True, max_iter=400)
# nn.fit(self.x_train, self.y_train)
# rand = RandomForestClassifier(n_jobs=200,verbose=1)
# rand.fit(self.x_train, self.y_train)

# ens["Grad"] = clf.predict(self.x_test)
# ens["Ext"] = bagging.predict(self.x_test)
# ens["NN"] = nn.predict(self.x_test)
# ens["Rand"] = rand.predict(self.x_test)

 
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



