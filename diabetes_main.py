import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrapers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)

# Transform and display the training data
X_standardized = scaler.transform(X)
data = pd.DataFrame(X_standardized)

def create_model:
    # create
    model = Seqential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim=8, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model_1 = create_model()
print(model_1.summary())

# Do a grid search for the optimal batch size and number of epochs
model_2 = KerasClassfier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 50, 100]

# make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=KFold(random_state=seed), verbose=10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(
    grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
