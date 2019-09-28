# Do a grid search to optimize kernel initialization and activation functions

# import necessary packages
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dropout
from diabetes_main import return_var

# Define a random seed
seed = 6
np.random.seed(seed)

X_standardized, y = return_var()
# Start defining the model
def create_model(activation, init):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer=init, activation=activation))
    model.add(Dense(4, input_dim=8, kernel_initializer=init, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model


# create the model
model = KerasClassifier(build_fn=create_model,
                        epochs=100, batch_size=20, verbose=0)

# define the grid search parameters
activation = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# make a dictionary of the grid search parameters
param_grid = dict(activation=activation, init=init)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=KFold(random_state=seed), verbose=10)
grid_results = grid.fit(X_standardized, y)

# summarize the results
print("Best: {0}, using {1}".format(
    grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
