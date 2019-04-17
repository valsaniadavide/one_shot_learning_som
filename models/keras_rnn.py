import numpy as np
np.set_printoptions(threshold=np.inf)
import logging
import glob
import pickle
from functools import reduce
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import SGD, RMSprop
from utils.constants import Constants


maximum_length = 1000

def build_keras_rnn():
    model = Sequential()
    model.add(LSTM(100, input_shape=(maximum_length, 1)))
    model.add(Dense(1000, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_keras_rnn(model):
    # load prepare the data
    #train_speakers_dataset, test_speaker_dataset = load_all_speakers()
    pickle_files = glob.glob('*-set.pickle')
    with open(pickle_files[1], 'rb') as train_file:
        train_speakers_dataset = pickle.load(train_file)
    with open(pickle_files[0], 'rb') as test_file:
        test_speaker_dataset = pickle.load(test_file)

    X_train = train_speakers_dataset.X
    y_train = train_speakers_dataset.y
    X_test = test_speaker_dataset.X
    y_test = test_speaker_dataset.y
    logging.debug('Total size of train dataset: ' + str(np.shape(X_train)))
    logging.debug('Total size of test dataset: ' + str(np.shape(X_test)))

    # old padding strategy. leaving it here for reference.
    #X_train_w = sequence.pad_sequences(X_train, maxlen=maximum_length)
    #X_test_w = sequence.pad_sequences(X_test, maxlen=maximum_length)

    # pad sequences
    X_temp = np.zeros((np.shape(X_train)[0], maximum_length))
    i = 0
    for x in X_train:
        X_temp[i] = x[0:maximum_length]
        i += 1
    X_train = X_temp

    X_temp = np.zeros((np.shape(X_test)[0], maximum_length))
    i = 0
    for x in X_test:
        X_temp[i] = x[0:maximum_length]
        i += 1
    X_test = X_temp


    logging.debug('Total size of train dataset: ' + str(np.shape(X_train)))
    logging.debug('Total size of test dataset: ' + str(np.shape(X_test)))

    X_train = np.reshape(X_train, (X_train.shape[0], maximum_length, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], maximum_length, 1))

    logging.debug('Total size of train dataset: ' + str(np.shape(X_train)))
    logging.debug('Total size of test dataset: ' + str(np.shape(X_test)))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)
    return model

def test_keras_rnn(trained_model):
    pass

def load_all_speakers():
    dataset_list = []
    for speaker in Constants.AVAILABLE_SPEAKERS:
        temp_dataset = OSXSpeakerDataset(speaker)
        temp_dataset.load()
        dataset_list.append(temp_dataset)
    return reduce((lambda x, y: x + y), dataset_list[:-1]), dataset_list[-1]


if __name__ == '__main__':
    np.random.seed(7)
    model = build_keras_rnn()
    trained_model = train_keras_rnn(model)
    model_path = 'adam-epochs100-batchsize64-seqlength1000.pickle'
    model.save(model_path)
    test_keras_rnn(trained_model)
