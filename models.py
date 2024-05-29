import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

filename = 'Pulsar.csv'
df_raw = pd.read_csv(filename, sep=',', encoding='utf-8')

X = df_raw.drop(["Class"], axis=1)
y = df_raw.loc[:, df_raw.columns == "Class" ]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, stratify=y, random_state=0) 

modelo_SVC = SVC( C=1, kernel= 'linear', probability=True, random_state=123)
modelo_SVC.fit(X_train, y_train)

filename_model = 'modelo.pkl'
with open(filename_model, 'wb') as archivo:
    pickle.dump(modelo_SVC, archivo)

print("Modelo generado!")




