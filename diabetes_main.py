# Author: Atheesh Krishnan
# 29th September, 2019
# Main Program | Driver Code

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from gscv_mark5 import final_gscv

# import the uci pima indians diabetes dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['n_pregnant', 'glucose_concentration', 'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)',
        'BMI', 'pedigree_function', 'age', 'class']

df = pd.read_csv(url, names = names)

columns = ['glucose_concentration',
           'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
dataset = df.values
print(dataset.shape)

X = dataset[:,0:8]
y = dataset[:,8].astype(int)

# Normalize the data using sklearn StandardScaler
scaler = StandardScaler().fit(X)

# Transform and display the training data
X_standardized = pd.DataFrame(scaler.transform(X))

def return_var():
    return X_standardized, y

# Importing from the gscv_mark5 script
grid = final_gscv()

y_pred = grid.predict(X_standardized)

# Model Results
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))