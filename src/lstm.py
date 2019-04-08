# *_*coding:utf-8 *_*
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, merge, Bidirectional
from keras.layers import LSTM, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import *


batch_size = 64
nb_feat = 34
nb_class = 4

optimizer = 'Adadelta'


def data_preparation(params):
    X, y, valid_idxs = get_sample(ids=None, take_all=True)
    y = to_categorical(y, params)
    idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

    X, X_mask = pad_sequence_into_array(X, maxlen=78)

    X_train, X_test = X[idxs_train], X[idxs_test]
    y_train, y_test = y[idxs_train], y[idxs_test]

    class_weights = np.unique(y, return_counts=True)[1] * 1.  # 每种情感多少个
    class_weights = np.sum(class_weights) / class_weights

    sample_weight = np.zeros(y_train.shape[0])
    for num, cls in enumerate(y_train):
        sample_weight[num] = class_weights[cls[0]]

    return X_train, y_train, X_test, y_test, sample_weight


def build_sequential_lstm(nb_feat, nb_class, optimizer='Adadelta'):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'),
                            merge_mode='concat', input_shape=(78, nb_feat)))
    model.add(Bidirectional(LSTM(64, return_sequences=False, activation='tanh'),
                            merge_mode='concat'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(nb_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_blstm(nb_feat, nb_class, optimizer='Adadelta'):
    net_input = Input(shape=(78, nb_feat))
    forward_lstm1 = LSTM(output_dim=64, return_sequences=True, activation='tanh')(net_input)
    backward_lstm1 = LSTM(output_dim=64, return_sequences=True, activation='tanh', go_backwards=True)(net_input)
    blstm_output1 = merge.concatenate([forward_lstm1, backward_lstm1], axis=-1)

    forward_lstm2 = LSTM(output_dim=64, return_sequences=False, activation='tanh')(blstm_output1)
    backward_lstm2 = LSTM(output_dim=64, return_sequences=False, activation='tanh', go_backwards=True)(blstm_output1)
    blstm_output2 = merge.concatenate([forward_lstm2, backward_lstm2], axis=-1)

    hidden = Dense(512, activation='tanh')(blstm_output2)
    output = Dense(nb_class, activation='softmax')(hidden)

    model = Model(net_input, output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


def train(model, X_train, y_train, sample_weight, batch_size):
    checkpoint_path = '../model/lstm_model/lstm_model_{epoch:03d}.hdf5'

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, period=50)
    earlystopping = EarlyStopping(monitor='acc', patience=10, mode='max')
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=150,
              verbose=1,
              sample_weight=sample_weight,
              callbacks=[checkpoint, earlystopping])

    return model


def predict(model, X_test, y_test):
    result = model.predict_classes(X_test)  #只有sequential模型才有predict_classes方法
    y_test = np.argmax(y_test, axis=1)

    ua = unweighted_accuracy(y_test.ravel(), result.ravel())
    wa = weighted_accuracy(y_test.ravel(), result.ravel())

    print('UA Test = {:.2%}, WA Test = {:.2%}'.format(ua, wa))


if __name__ == '__main__':
    start = time.time()
    params = Constants()
    print(params)

    model = build_sequential_lstm(nb_feat=nb_feat, nb_class=nb_class, optimizer=optimizer)
    model.summary()

    X_train, y_train, X_test, y_test, sample_weight = data_preparation(params)

    model = train(model, X_train, y_train, sample_weight, batch_size)

    predict(model, X_test, y_test)

    end = time.time()
    print('总用时：{:.2f}mins'.format((end-start)/60))
