import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1, 2])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100
)

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(cm)
print("Accuracy:", accuracy)

single_prediction = ann.predict(
    sc.transform(
        [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
    )
)

print(single_prediction > 0.5)
