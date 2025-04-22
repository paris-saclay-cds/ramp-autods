from sklearn.base import BaseEstimator                                                                  
from sklearn.svm import SVR                                                                           
from sklearn.pipeline import make_pipeline                                                              
from sklearn.preprocessing import StandardScaler                                                        
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
c = Hyperparameter(dtype='float', default=2 ** 11, values=[2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15])
gamma = Hyperparameter(dtype='float', default=2 ** -15, values=[2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3])
# RAMP END HYPERPARAMETERS# RAMP END HYPERPARAMETERS                                                                              

C = float(c)
GAMMA = float(gamma)

class Regressor(BaseEstimator):                                                                        
    def __init__(self, metadata):                                                                    
        self.metadata = metadata
                                                                                                        
    def fit(self, X, y):                                                                                
        if self.metadata["score_name"] == "rmsle":
            y = np.log1p(y)
        self.reg = make_pipeline(                                                                       
            StandardScaler(),                                                                           
            SVR(                                                                                        
                C=C,                                                                             
                gamma=GAMMA,                                                                     
            ))                                                                                          
        self.reg.fit(X, y)                                                                              
                                                                                                        
    def predict(self, X):
        y_pred = self.reg.predict(X)
        if self.metadata["score_name"] == "rmsle":
            y_pred = np.expm1(y_pred)
        return y_pred
