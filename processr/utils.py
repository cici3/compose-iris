import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
#classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
classes = {0: "Class_0", 1: "Class_1", 2: "Class_2"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    # processed = [
    #     {
    #         "sepal_length": d.sepal_length,
    #         "sepal_width": d.sepal_length,
    #         "petal_length": d.petal_length,
    #         "petal_width": d.petal_width,
    #         "flower_class": d.flower_class,
    #     }
    #     for d in data
    # ]

    processed = [
        {
            "Alcohol": d.Alcohol,
            "Malic_acid": d.Malic_acid,
            "Ash": d.Ash,
            "Alcalinity_of_ash": d.Alcalinity_of_ash,
            "Magnesium": d.Magnesium,
            "Total_phenols": d.Total_phenols,
            "Flavanoids": d.Flavanoids,
            "Nonflavanoid_phenols": d.Nonflavanoid_phenols,
            "Proanthocyanins": d.Proanthocyanins,
            "Color_intensity": d.Color_intensity,
            "Hue": d.Hue,
            "OD280_OD315_of_diluted_wines": d.OD280_OD315_of_diluted_wines,
            "wine_class": d.wine_class
         
        }
        for d in data
    ]

    return processed


 