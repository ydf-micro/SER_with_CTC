# -*- coding: utf-8 -*-
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, merge
from keras.layers import LSTM, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics

from utils import *

batch_size = 64
nb_feat = 34
nb_class = 4

optimizer = 'Adadelta'


def get_feature(params):

    params.path_to_data = '/home/ydf_micro/datasets/IEMOCAP_full_release/'

    start = time.time()
    data = read_iemocap_data(params=params)
    end = time.time()
    print('运行时间：{:.2f}s'.format((end-start)/60))

    start = time.time()
    get_features(data, params)
    end = time.time()
    print('运行时间：{:.2f}s'.format((end-start)/60))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(nb_feat, nb_class, optimizer='Adadelta'):
    net_input = Input(name='the_input', shape=(78, nb_feat))
    forward_lstm1 = LSTM(output_dim=64, return_sequences=True, activation='tanh')(net_input)
    backward_lstm1 = LSTM(output_dim=64, return_sequences=True, activation='tanh', go_backwards=True)(net_input)
    blstm_output1 = merge.concatenate([forward_lstm1, backward_lstm1], axis=-1)

    forward_lstm2 = LSTM(output_dim=64, return_sequences=True, activation='tanh')(blstm_output1)
    backward_lstm2 = LSTM(output_dim=64, return_sequences=True, activation='tanh', go_backwards=True)(blstm_output1)
    blstm_output2 = merge.concatenate([forward_lstm2, backward_lstm2], axis=-1)

    hidden = TimeDistributed(Dense(512, activation='tanh'))(blstm_output2)
    output = TimeDistributed(Dense(nb_class+1, activation='softmax'))(hidden)

    labels = Input(name='the_labels', shape=[1], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])

    model = Model(input=[net_input, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer,
                  metrics=['acc'])

    test_func = K.function([net_input], [output])

    return model, test_func


def data_preparation(params):
    X, y, valid_idxs = get_sample(ids=None, take_all=True)
    y = np.argmax(to_categorical(y, params), axis=1)
    y = np.reshape(y, (y.shape[0], 1))

    X, X_mask = pad_sequence_into_array(X, maxlen=78)
    y, y_mask = pad_sequence_into_array(y, maxlen=1)

    index_to_retain = np.sum(X_mask, axis=1, dtype=np.int32) > 5

    X, X_mask = X[index_to_retain], X_mask[index_to_retain]
    y, y_mask = y[index_to_retain], y_mask[index_to_retain]

    idxs_train, idxs_test = train_test_split(range(X.shape[0]))
    X_train, X_test = X[idxs_train], X[idxs_test]
    X_train_mask, X_test_mask = X_mask[idxs_train], X_mask[idxs_test]
    y_train, y_test = y[idxs_train], y[idxs_test]
    y_train_mask, y_test_mask = y_mask[idxs_train], y_mask[idxs_test]

    return X, y, X_train, X_test, X_train_mask, X_test_mask, y_train, y_test, y_train_mask, y_test_mask

def UA_WA(X, y, X_mask, y_mask, sess, test_func):
    inputs = {'the_input': X,
              'the_labels': y,
              'input_length': np.sum(X_mask, axis=1, dtype=np.int32),
              'label_length': np.squeeze(y_mask),
              }
    preds = test_func([inputs["the_input"]])[0]
    decode_function = K.ctc_decode(preds[:, 2:, :], inputs["input_length"] - 2, greedy=False, top_paths=1)
    labellings = decode_function[0][0].eval(session=sess)
    if labellings.shape[1] == 0:
        ua = 0.0
        wa = 0.0
    else:
        ua = unweighted_accuracy(y.ravel(), labellings.T[0].ravel())
        wa = weighted_accuracy(y.ravel(), labellings.T[0].ravel())

    # loss = np.mean(model.predict(inputs))

    return ua, wa


def train(model, test_func, X, y, X_train, X_test, X_train_mask,
          X_test_mask, y_train, y_test, y_train_mask, y_test_mask):
    checkpoint_path = '../model/ctc_model/ctc_model_{epoch:03d}.hdf5'
    sess = tf.Session()

    class_weights = np.unique(y, return_counts=True)[1] * 1.  # 每种情感多少个
    class_weights = np.sum(class_weights) / class_weights

    sample_weight = np.zeros(y_train.shape[0])
    for num, cls in enumerate(y_train):
        sample_weight[num] = class_weights[cls[0]]

    def generator_data(X_train, y_train, X_train_mask, y_train_mask, sample_weight, batch_size):
        '''
            这里必须写成了一个死循环，因为model.fit_generator()在使用在个函数的时候，
            并不会在每一个epoch之后重新调用，那么如果这时候generator自己结束了就会有问题。
        '''
        while True:
            batches = range(0, (X_train.shape[0] // batch_size) * batch_size, batch_size)
            shuffle = np.random.choice(batches, size=len(batches), replace=False)
            for num, i in enumerate(shuffle):
                inputs_train = {'the_input': X_train[i:i + batch_size],
                                'the_labels': y_train[i:i + batch_size],
                                'input_length': np.sum(X_train_mask[i:i + batch_size], axis=1, dtype=np.int32),
                                'label_length': np.squeeze(y_train_mask[i:i + batch_size])}
                outputs_train = {'ctc': np.zeros([inputs_train["the_labels"].shape[0]])}

                yield (inputs_train, outputs_train, sample_weight[i:i + batch_size])

    batch = generator_data(X_train, y_train, X_train_mask, y_train_mask, sample_weight, batch_size)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, period=50)
    earlystopping = EarlyStopping(monitor='acc', patience=10, mode='max')
    model.fit_generator(batch, steps_per_epoch=(X_train.shape[0]//batch_size), epochs=500,
                        callbacks=[checkpoint, earlystopping])

    ua_train, wa_train = UA_WA(X_train, y_train, X_train_mask, y_train_mask, sess, test_func)
    ua_test, wa_test = UA_WA(X_test, y_test, X_test_mask, y_test_mask, sess, test_func)

    print('UA Train = {:.2%}, WA Train = {:.2%}, UA Test = {:.2%}, WA Test = {:.2%}'.format(ua_train,
                                                                                            wa_train,
                                                                                            ua_test,
                                                                                            wa_test))

    sess.close()


if __name__ == '__main__':
    start = time.time()
    params = Constants()
    print(params)
    model, test_func = build_model(nb_feat=nb_feat, nb_class=nb_class, optimizer=optimizer)
    model.summary()
    X, y, X_train, X_test, X_train_mask, \
    X_test_mask, y_train, y_test, y_train_mask, y_test_mask = data_preparation(params)
    train(model, test_func, X, y, X_train, X_test, X_train_mask,
          X_test_mask, y_train, y_test, y_train_mask, y_test_mask)
    end = time.time()
    print('总用时：{:.2f}mins'.format((end-start)/60))
