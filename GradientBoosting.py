import numpy as np
import os 
from sklearn.tree import DecisionTreeRegressor
os.system("cls")

class Gradient_Boosting_Regressor :
    def __init__(self , num_of_models = 100, learning_rate = 0.1 ,max_depth = 2 ):
        self.num_of_models = num_of_models
        self.learning_rate = learning_rate      # fixeed for all models 
        self.models = []     # list contsins the models we used  
        self.max_depth = max_depth
        self.y = None
    def fit (self , X , y):
        self.y = y
        initial_predetion = np.mean(y)      # mean for y 
        y_hat = np.ones_like (y) * initial_predetion   # mean as vector for all data point 

        for _ in range (self.num_of_models):
            error = y - y_hat
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X,error)      
            predicted_error = model.predict(X)                 # predict error
            y_hat+= self.learning_rate*predicted_error         # y hat (we want to predict) 
            self.models.append(model)    

    def predict(self , X):    # to predict new data point 
        # sum of initial prediction (mean) and error of each model multiply of learning rate 
        y_hat = np.mean(self.y)    # initail value 
        for model in self.models:
            y_hat += self.learning_rate * model.predict(X)
        return y_hat

