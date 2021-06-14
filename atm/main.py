import sqlite3 as sql
from os.path import dirname, join

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.plotting import figure

atms = pd.read_csv('dataset_atm_01.03.txt', sep='\t')
atms.iloc[:,-1]=atms.iloc[:,-1].str.replace(',','.')
atms.iloc[:,-1].replace('#Н/Д', np.nan, inplace=True)
atms.iloc[:, -1] = atms.iloc[:,-1].astype(np.float)
atms.fillna(0, inplace=True)

label_encoder = LabelEncoder()
text_features = ['Статус']
for col in text_features:
    atms[col] = label_encoder.fit_transform(atms[col])

atms.drop( atms[ atms['Статус'] == 0 ].index , inplace=True)

atms["color"] = np.where(atms["Статус"] == 1, "green", "red")

axis_map = {
    "Статус": "Статус",
    "Год": "Год",
    "Время просрочки, ч": "Время просрочки, ч",
    "Чековый принтер--Сбой": "Чековый принтер--Сбой",
    "Программное обеспечение--OFFLINE. Требуется перезагрузка": "Программное обеспечение--OFFLINE. Требуется перезагрузка"
}

desc = Div(text=open(join(dirname(""), "description.html")).read(), sizing_mode="stretch_width")

# Create Input controls
status = Slider(title="Показывать снятые", value=0, start=0, end=1, step=1)
year = Slider(title="Год", start=2018, end=2020, value=2018, step=1)
delay_time = Slider(title="Время просрочки", start=0, end=500, value=0, step=10)
last_task_date = TextInput(title="Дата последней задачи")
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Статус")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Время просрочки, ч")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], year=[]))

TOOLTIPS=[
    ("Year", "@year")
]

p = figure(height=600, width=700, title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None)


def select_atms():
    last_task_date_val = last_task_date.value.strip()
    selected = atms[
        (atms['Статус'] <= status.value + 1) &
        (atms['Год'] <= year.value ) &
        (atms['Время просрочки, ч'] >= delay_time.value)
    ]
    if (last_task_date_val != ""):
        selected = selected[selected['Дата последней задачи'].str.contains(last_task_date_val)==True]
    return selected


def update():
    df = select_atms()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d atms selected" % len(df)
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        year=df["Год"]
    )

controls = [status, year, delay_time, last_task_date, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(*controls, width=320)

l = column(desc, row(inputs, p), sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Atms"