import matplotlib.pyplot as plt
import seaborn as sns 
from postprocessing import Postprocessing 
import svm
import pandas as pd
from pandas.plotting import radviz


class Visualize():
    # read the data
    
    
#  'f_a_0', 'f_a_1', 'f_a_2', 'f_a_3', 'f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4'] 'ind_0','ind_1','ind_2','ind_3','ind_4'
    
    
    def plot_features(training_data):
 
        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        X = x_train
        y = y_train
        plt.scatter(X['ind_0'],X['ind_1'], c=y, s=30,cmap='seismic')
        plt.scatter(X['ind_2'],X['ind_3'], c=y, s=30,cmap='seismic')
        plt.scatter(X['ind_4'], c=y, s=30,cmap='seismic')
        plt.show()
        

if __name__ == "__main__":
    
    training_data = pd.read_csv('data_set_v2.csv' , index_col=0)
    pd.plotting.scatter_matrix(training_data[['ind_0','ind_1','ind_2','ind_3']], alpha=0.2)
    plt.show()
    # Visualize.plot_features(training_data)
    