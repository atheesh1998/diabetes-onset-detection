import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrapers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

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
X_standardized = scaler.transform(X)
data = pd.DataFrame(X_standardized)
