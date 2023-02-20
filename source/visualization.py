import matplotlib.pyplot as plt
import seaborn as sns 
from postprocessing import Postprocessing 
import svm
import pandas as pd



class Visualize():
    # read the data
    
    
    
    
    
    def plot_features(training_data):
 
        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        X = x_train.values
        y = y_train
        plt.scatter(X[:, 2], X[:, 3], c=y, s=30,cmap='seismic')
        plt.show()
        

if __name__ == "__main__":
    
    training_data = svm.Prediction.data_frames(pd.read_csv('data_set_v1.csv' , index_col=False))
    Visualize.plot_features(training_data)
    