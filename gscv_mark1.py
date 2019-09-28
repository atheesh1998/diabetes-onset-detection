



def create_model:
    # create
    model = Seqential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile
    adam = Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model


model_1 = create_model()
print(model_1.summary())
