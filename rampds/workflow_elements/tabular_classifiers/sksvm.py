from sklearn.base import BaseEstimator                                                                  
from sklearn.svm import SVC                                                                             
from sklearn.pipeline import make_pipeline                                                              
from sklearn.preprocessing import StandardScaler                                                        
from ramphy import Hyperparameter                                                              
                                                                                                        
                                                                                                        
# RAMP START HYPERPARAMETERS                                                                            
# RAMP START HYPERPARAMETERS
c = Hyperparameter(dtype='float', default=2 ** 11, values=[2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15])
gamma = Hyperparameter(dtype='float', default=2 ** -15, values=[2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3])
# RAMP END HYPERPARAMETERS# RAMP END HYPERPARAMETERS                                                                              

C = float(c)
GAMMA = float(gamma)

class Classifier(BaseEstimator):                                                                        
    def __init__(self, dtypes_dict):                                                                    
        pass                                                                                            
                                                                                                        
    def fit(self, X, y):                                                                                
        self.clf = make_pipeline(                                                                       
            StandardScaler(),                                                                           
            SVC(                                                                                        
                C=C,                                                                             
                gamma=GAMMA,                                                                     
                probability=True                                                                        
            ))                                                                                          
        self.clf.fit(X, y)                                                                              
                                                                                                        
    def predict_proba(self, X):                                                                         
        return self.clf.predict_proba(X)                                                                
