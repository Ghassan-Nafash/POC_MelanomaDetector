import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error , matthews_corrcoef
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import time
from utilities import Utilities
import cv2
from preprocessing import Preprocessing 
from postprocessing import Postprocessing
from segmentation import NormalizedOtsuWithAdaptiveThresholding
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Prediction():
    """
    this methods takes the generated features as input
    """
    
    

    
    def data_frames(data_frames):
        global target_vector
        df_feat = data_frames[['f_a_0', 'f_a_1', 'f_a_2', 'f_a_3', 'f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4']]

        # df_feat = data_frames[['ind_0','ind_1','ind_2','ind_3','ind_4']]
        
        df_target = data_frames['metadata_label']
        # df_feat = df_feat[['f_a_0', 'f_a_1', 'f_a_2', 'f_a_3']]
        # df_feat = df_feat[['f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4']]
       
        
        print("df_feature_independent", df_feat)
        target_vector = np.ravel(df_target)                
        
        x_train, x_test, y_train, y_test = train_test_split(df_feat, target_vector, test_size=0.20,random_state=101)

        # print("x_train_", x_train)

        normalized_x, x_mean , x_std = Prediction.normalize_data(x_train)


        return [normalized_x, x_test, y_train, y_test, x_mean, x_std, df_feat,df_target]
    

    def normalize_data(data):
        mean_list = []
        std_deviation_list = [] 

        for feature in data:        

            # calculate the mean and standard deviation of the data
            mean = np.mean(data[feature])

            mean_list.append(mean)

            std_deviation = np.std(data[feature])

            std_deviation_list.append(std_deviation)

            # normalize the data using the formula
            feature_normalized = (data[feature] - mean) / std_deviation

            #data_normalized_list.append(feature_normalized)
            data[feature] = feature_normalized

        
        return data, mean_list, std_deviation_list


    def normalize_data_for_prediction(data, mean_list, std_deviation_list):

            index = 0
            for feature in data:

                # normalize the data using the formula
                feature_normalized = (data[feature] - mean_list[index]) / std_deviation_list[index]

                data[feature] = feature_normalized

                index += 1
            

            return data

    def grid_search_RBF(training_data:list):
        global target_vector
        #grid.best_params_= {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}

        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        x_mean = training_data[4]
        x_std = training_data[5]
        df_feat = training_data[6]
        df_target = training_data[7]

        # poly = PolynomialFeatures(degree=4, include_bias=True)
        # poly_features = poly.fit_transform(df_feat)
        # poly_X_train, poly_x_test, poly_y_train,poly_y_test = train_test_split(poly_features,df_target,test_size=0.3,random_state=101)
        # poly_reg_model = LinearRegression()
        # poly_reg_model.fit(poly_X_train, poly_y_train)
        
        # poly_reg_y_predicted = poly_reg_model.predict(poly_x_test)
        

        
        # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        
        # param_grid = {'C': [0.01, 0,1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear','rbf']}
        
        # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        svm_rbf = SVC(kernel='rbf', C=1, gamma=0.01)

        svm_rbf.fit(x_train, y_train)
        
        normalized_test_data = Prediction.normalize_data_for_prediction(x_test, x_mean, x_std)

        svm_rbf_predictions = svm_rbf.predict(normalized_test_data)

        # print("grid.best_params_=",grid.best_params_)
        # print("grid.best_score_=",grid.best_score_)

        print(confusion_matrix(y_test, svm_rbf_predictions))

        print(classification_report(y_test, svm_rbf_predictions))
        
        MSE = mean_squared_error(y_test, svm_rbf_predictions)
        print("MSE=", MSE)
        MCC = matthews_corrcoef(y_test,svm_rbf_predictions) 
        print("MCC= ",MCC)
        
        # poly_reg_rmse = np.sqrt(mean_squared_error(poly_y_test, poly_reg_y_predicted))
        # print("Poly Reg Model: ",poly_reg_rmse)
        
        
        return svm_rbf_predictions


    def grid_search(training_data:list):        
        
        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        x_mean = training_data[4]
        x_std = training_data[5]

        print("NaNs_after_normalization", x_train.isnull().sum().sum())
        
        "find the optimal param for the model"
        #param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        
        RegModel = SVC(C=1.0, kernel='linear')
 
        #Printing all the parameters of KNN
        print(RegModel)
        
        #Creating the model on Training Data
        SVM_var = RegModel.fit(x_train, y_train)                

        normalized_test_data = Prediction.normalize_data_for_prediction(x_test, x_mean, x_std)

        prediction = SVM_var.predict(normalized_test_data)
        
        MSE = mean_squared_error(y_test, prediction)

        print("MSE=", MSE)

        #print("prediction=", prediction)

        #plottingSVM.plot_svm_boundary(RegModel, x_train, y_test)

        #Measuring Goodness of fit in Training data
        #from sklearn import metrics
        #print('R2 Value:',metrics.r2_score(y_test, SVM.predict(x_test)))
        
        #Measuring accuracy on Testing Data
        #print('Accuracy',100- (np.mean(np.abs((x_test - prediction) / y_test)) * 100))

        print(confusion_matrix(y_test, prediction))

        print(classification_report(y_test, prediction))
        
    ################
    
    
""" def generate_data(features):
            
    
        img_feature_list = { 'img_number': img_number,
                            
                            'metadata_label': meta_data[img_number], # 1 for malign etc. positive, 0 for benign etc. negative

                            'f_a_0':features[0][0],    
                            'f_a_1':features[0][1], 
                            'f_a_2':features[0][2], 
                            'f_a_3':features[0][3],

                            'f_b_0':features[1],

                            'f_c_0':features[2][0],
                            'f_c_1':features[2][1],
                            'f_c_2':features[2][2],
                            'f_c_3':features[2][3],
                            'f_c_4':features[2][4],
                            
                            'ind_0':independent_features[0],
                            'ind_1':independent_features[1],
                            'ind_2':independent_features[2],
                            'ind_3':independent_features[3],
                            'ind_4':independent_features[4]
                                                        
                            }
        if (None in img_feature_list.values()): img_failed += 1

        data_set.append(img_feature_list)

        Utilities.save_dataset(dataset=data_set, file_path="./data_set_v2.csv", only_succesfull=True)

        end_time = time.process_time()

        total_time = (end_time - start_time)*1000 # in millis

        avg_time = total_time / img_count
        
        print("total_time = %.0f min" % (total_time/1000/60))
        print("avg_time = %.0f ms per image" % avg_time)
        print("img_failed: %d ... %.1f%% of total images" %(img_failed, img_failed/img_count*100))
        print("img_count", img_count)"""
