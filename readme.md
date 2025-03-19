# self replicating NN


# layers

## Dense

Dense(32, activation='selu')

## Convolutional Layers

Conv2D(filters=32, kernel_size=(3, 3), activation='relu')


## Pooling Layers

MaxPooling2D(pool_size=(2, 2))


## Recurrent Layers

LSTM(32, activation='tanh', return_sequences=True)


## Dropout Layers

Dropout(0.5)  # 50% neurons randomly dropped during training


## Batch Normalization Layers

BatchNormalization()


## Activation functions

* relu
* sigmoid
* tanh
* LeakyReLU
* softmax
* linear
* elu
* selu

## Amount of neurons in each layer


