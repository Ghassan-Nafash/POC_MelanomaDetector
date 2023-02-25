#!/usr/bin/env python3

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from utilities import Utilities
import pickle
from algorithm import ProcessingAlgorithm


class Prediction():
    """
    these methods takes the generated features as input
    """
    def data_frames(data_frames):
        
        global target_vector

        df_feat = data_frames[['f_a_0', 'f_a_1', 'f_a_2', 'f_a_3', 'f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4']]

        #df_feat = data_frames[['ind_0','ind_1','ind_2','ind_3','ind_4']]
        
        df_target = data_frames['metadata_label']
                       
        target_vector = np.ravel(df_target)                
        
        x_train, x_test, y_train, y_test = train_test_split(df_feat, target_vector, test_size=0.20, random_state=101)

        normalized_x, x_mean , x_std = Prediction.normalize_data(x_train)

        return [normalized_x, x_test, y_train, y_test, x_mean, x_std, df_feat, df_target]
    

    def balance_dataset(complete_dataset_frame):
        """
        This method will delete number of rows from original dataset to balance the number
        of positive and negative samples for training. 
        
        input: original dataset as pandas frame
        output: pd balanced dataset
        """
        
        groups = complete_dataset_frame.groupby('metadata_label')

        min_size = groups.size().min()

        out = groups.apply(lambda g: g.sample(min(len(g), min_size)))

        return out


    def normalize_data(data):
        '''
        normalize training data set
        '''
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
            '''
            normalize test data set using mean and variance as input
            from training data set
            '''

            index = 0
            for feature in data:

                # normalize the data using the formula
                feature_normalized = (data[feature] - mean_list[index]) / std_deviation_list[index]

                data[feature] = feature_normalized

                index += 1
            

            return data

    def train_classifier(training_data:list):
        '''
        applying grid search, to find best parameters of the kernel function
        '''
        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        x_mean = training_data[4]
        x_std = training_data[5]
        df_feat = training_data[6]
        df_target = training_data[7]

        #print(" x_train=",  x_train)

        # {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
        #param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}

        #grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        # best values after applying grid search
        svm_rbf = SVC(kernel='rbf', C=1000, gamma=0.1)

        svm_rbf.fit(x_train, y_train)
                
        normalized_test_data = Prediction.normalize_data_for_prediction(x_test, x_mean, x_std)

        grid_predictions = svm_rbf.predict(normalized_test_data)

        #print("grid.best_params_=",grid.best_params_)

        print(confusion_matrix(y_test, grid_predictions))

        print(classification_report(y_test, grid_predictions))

        MSE = mean_squared_error(y_test, grid_predictions)

        print("MSE=", MSE)
        

        return svm_rbf


    def run_svm(generated_features_path: str):
        '''
        find best suport vectors
        '''
        load_data_set = pd.read_csv(generated_features_path , index_col=0)

        balanced_data_set = Prediction.balance_dataset(load_data_set)

        training_data = Prediction.data_frames(balanced_data_set)

        #print("training_data=", training_data)

        x_mean = training_data[4]

        x_std = training_data[5]
        
        svm_rbf = Prediction.train_classifier(training_data)

        Prediction.save_classifier(svm_rbf, x_mean, x_std)
    

    def predict(data_set_path: str, image_number: int):
        correctness = True

        # load classifier
        classifier, mean, variance = Prediction.load_classifier('classifier.pkl')

        # processing image
        image_dict = Utilities.load_images_in_range(data_set_path, range_start=image_number, range_end=image_number + 1)
        
        features, independent_features = ProcessingAlgorithm.process_image(image_dict[image_number], image_number)

        img_feature_dict = {    'f_a_0':features[0][0],    
                                'f_a_1':features[0][1], 
                                'f_a_2':features[0][2], 
                                'f_a_3':features[0][3],
                                'f_b_0':features[1],
                                'f_c_0':features[2][0],
                                'f_c_1':features[2][1],
                                'f_c_2':features[2][2],
                                'f_c_3':features[2][3],
                                'f_c_4':features[2][4]                                                            
                                }

        # normalize data for prediction
        normalized_image_feature = Prediction.normalize_data_for_prediction(img_feature_dict, mean, variance)

        data_frame_from_list = pd.DataFrame([normalized_image_feature])

        grid_predictions = classifier.predict(data_frame_from_list)

        print("grid_predictions=", grid_predictions)

        return grid_predictions, correctness
        

    def save_classifier(classifier, mean, variance):        
        # Store the variables in a dictionary
        data = {'classifier': classifier, 'mean': mean, 'variance': variance}

        # Save the dictionary using pickle
        with open('classifier.pkl', 'wb') as file:
            pickle.dump(data, file)         


    def load_classifier(path: str):
        # Load the variables from the file
        with open(path, 'rb') as file:
            data = pickle.load(file)

            # Retrieve the variables from the loaded data
            classifier = data['classifier']
            mean = data['mean']
            variance = data['variance']

        return  classifier, mean, variance
