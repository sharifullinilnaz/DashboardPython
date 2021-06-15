from os.path import dirname, join

import pandas as pd
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn)

atms = pd.read_csv('dataset_atm_01.03.txt', sep='\t')
df = pd.read_csv('dataset_atm_01.03.txt', sep='\t')
result = pd.read_csv('dataset_atm_01.03.txt', sep='\t')

df.pop('IdATM')
df.pop('Год')
from sklearn.metrics import roc_curve, precision_recall_curve, auc # метрики качества
from sklearn.metrics import confusion_matrix, accuracy_score # метрики качества
from sklearn.metrics import average_precision_score # метрики качества
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
text_features = ['Статус']
for col in text_features:
   df[col] = label_encoder.fit_transform(df[col])
df.loc[(df['Статус'] == 1), 'Статус'] = 0
df.loc[(df['Статус'] == 2), 'Статус'] = 1
df.iloc[:,-1]=df.iloc[:,-1].str.replace(',','.')
df.iloc[:,-1].replace('#Н/Д', np.nan, inplace=True)
df.iloc[:, -1] = df.iloc[:,-1].astype(np.float)
df.iloc[:,-1].fillna(0, inplace=True)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)
train1 = train._get_numeric_data()
target_variable_name = 'Статус'
training_values = train1[target_variable_name]
training_points = train1.drop(target_variable_name, axis=1)
import pandas as pd
import numpy as np
import datetime
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras import backend as Kback

def auc(value_true, value_pred):
    auc = tf.metrics.auc(value_true, value_pred)[1]
    Kback.get_session().run(tf.local_variables_initializer())
    return auc

from keras.models import load_model

dependencies = {
    'auc': auc
}

model = keras.models.load_model('my_model2', custom_objects=dependencies)

df1 = df._get_numeric_data()
df1_points = df1.drop(target_variable_name, axis=1)
prediction = model.predict(df1_points)
result = result.join(pd.DataFrame(prediction, columns = ['model_result']))


source = ColumnDataSource(data=dict())

def update():
    current = result[(result['model_result'] >= slider.value[0]) & (result['model_result'] <= slider.value[1])].dropna()
    source.data = {
        'IdATM'             : current['IdATM'],
        'Статус'           : current['Статус'],
        'model_result' : current['model_result']
    }

slider = RangeSlider(title="Min вероятность", start=0, end=1, value=(0, 0.1), step=0.005)
slider.on_change('value', lambda attr, old, new: update())


columns = [
    TableColumn(field="IdATM", title="IdATM"),
    TableColumn(field="Статус", title="Статус"),
    TableColumn(field="model_result", title="Вероятность")
]

data_table = DataTable(source=source, columns=columns, width=800)

controls = column(slider)

curdoc().add_root(row(controls, data_table))
curdoc().title = "Atm"

update()