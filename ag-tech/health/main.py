import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

dataframe = pd.read_excel('../input/cattle_dataset.xlsx')
label_encoder = LabelEncoder()


def data_preprocessing(dataframe, column_name_list):
    for index in column_name_list:
        dataframe[index] = label_encoder.fit_transform(dataframe[index])
        # print(dataframe[index])


data_preprocessing(dataframe, ['faecal_consistency', 'health_status', 'breed_type'])
independent_X = dataframe[
    [
        'body_temperature', 'breed_type', 'milk_production', 'respiratory_rate', 'walking_capacity',
        'sleeping_duration',
        'body_condition_score', 'heart_rate', 'eating_duration', 'lying_down_duration', 'ruminating', 'rumen_fill',
        'faecal_consistency'
    ]
]
dependent_y = dataframe['health_status']
# print('Independent X =\n', independent_X['body_temperature'])
# print('Dependent Y =\n', dependent_y)

X_train, X_test, y_train, y_test = train_test_split(independent_X, dependent_y, test_size=0.20)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.ravel()
y_test = y_test.ravel()
# print("X_train=", X_train.shape)
# print("y_train=", y_train.shape)
# print("X_test=", X_test.shape)
# print("y_test=", y_test.shape)

cattle_classifier = SVC(probability=True, kernel='rbf')
cattle_classifier.fit(X_train, y_train)
prediction = cattle_classifier.predict(X_test)
print('Prediction => ', prediction)
# print("MAE=%.4f" % mean_absolute_error(y_test, prediction))
# print("MSE=%.4f" % mean_squared_error(y_test, prediction))
result = cattle_classifier.score(X_test, y_test)
# print('Score=', result)

label_encoder.fit_transform(dataframe['health_status'])
original_health_status_list = list(dataframe['health_status'])
ids = list(dataframe['id'])
print('Current health status => ', original_health_status_list)

data_preprocessing(dataframe, ['faecal_consistency', 'breed_type'])
X_test = dataframe[
    ['body_temperature', 'breed_type', 'milk_production', 'respiratory_rate', 'walking_capacity', 'sleeping_duration',
     'body_condition_score', 'heart_rate', 'eating_duration', 'lying_down_duration', 'ruminating', 'rumen_fill',
     'faecal_consistency']]
X_test = X_test.to_numpy()
prediction = cattle_classifier.predict(X_test)
predicted_health_status_list = []
for i in prediction:
    data = label_encoder.inverse_transform([i])
    predicted_health_status_list.append(label_encoder.inverse_transform([i]))

column_name = ['ID', 'Original Output', 'Prediction Output']
result_df = pd.DataFrame(
    {'ID': ids, 'Original Output': original_health_status_list, 'Prediction Output': predicted_health_status_list},
    columns=column_name)
result_df.to_csv('./ResultCattle.csv')