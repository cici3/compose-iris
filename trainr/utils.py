import os
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split

# define the class encodings and reverse encodings
#classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
classes = {0: "Class_0", 1: "Class_1", 2: "Class_2"}
r_classes = {y: x for x, y in classes.items()}

# def train_model_first():
#     wine = load_wine() 
#     df = pd.DataFrame(wine.data)
#     df['wine_class'] = wine.target
#     y = df['wine_class'].values
#     X = df.drop(['wine_class'],axis=1).values
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#     model = GaussianNB()
#     model.fit(X_train,y_train)
#     with open ("models/wine_nb.pkl", 'wb') as f:
#         pickle.dump(model,f)


# function to train and load the model during startup
def init_model():
    if not os.path.isfile("models/wine_nb.pkl"):
        clf = GaussianNB()
        pickle.dump(clf, open("models/wine_nb.pkl", "wb"))
 
# function to train and save the model as part of the feedback loop
def train_model(data):
    # load the model
    clf = pickle.load(open("models/wine_nb.pkl", "rb"))

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.wine_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    pickle.dump(clf, open("models/wine_nb.pkl", "wb"))
