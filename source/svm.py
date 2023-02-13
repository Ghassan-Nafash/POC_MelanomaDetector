import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

class Prediction():
    """
    this methods takes the generated features as input
    """
    def data_frames(data_frames):

        print(data_frames)
        
        #df_feat = pd.DataFrame(data_frames['img_number'], columns=data_frames['f_a_0', 'f_a_1', 'f_a_2', 'f_a_3', 'f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4'])
        
        #df_feat = pd.read_csv(index_col=False)
        df_feat = data_frames[['f_a_0', 'f_a_1', 'f_a_2', 'f_a_3', 'f_b_0', 'f_c_0', 'f_c_1', 'f_c_2', 'f_c_3', 'f_c_4']]

        #df_target = pd.DataFrame(data_frames['target'],columns=['metadata_label'])
        df_target = data_frames['metadata_label']        

        print("df_target", df_target)

        x_train, x_test, y_train, y_test = train_test_split(df_feat, df_target, test_size=0.30, random_state=101)

        return [x_train, x_test, y_train, y_test]
    
    def grid_search(training_data:list):        

        x_train = training_data[0]
        x_test = training_data[1]
        y_train = training_data[2]
        y_test = training_data[3]
        
        "find the optimal param for the model"
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        grid.fit(x_train, y_train)

        grid_predictions = grid.predict(x_test)

        print(confusion_matrix(y_test,grid_predictions))

        print(classification_report(y_test,grid_predictions))

        return grid_predictions
    