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
        df_feat = pd.DataFrame(data_frames['data'], columns=data_frames['feature_vector'])
        df_target = pd.DataFrame(data_frames['target'],columns=['Melanoma'])
        x_train, x_test, y_train, y_test = train_test_split(df_feat, test_size=0.30, random_state=101)
        return [x_train,x_test,y_train,y_test]
    
    def grid_search(x_train,x_test,y_train,y_test):
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        grid.fit(x_train, y_train)
        grid_predictions = grid.predict(x_test)
        return grid_predictions
        




