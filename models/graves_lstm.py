import tensorflow as tf
import librosa as lr
import pickle
import glob
import time
import numpy as np
import logging
import random
import os
from utils.utils import array_to_sparse_tuple_1d, array_to_sparse_tuple, get_next_batch_index
from utils.utils import pad_np_arrays
from utils.constants import Constants
from data.dataset import TIMITDataset
from keras.layers import Input, Bidirectional, LSTM
from keras.models import Sequential, Model
from keras.utils import to_categorical
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell


LOAD_PICKLE = True
# setting those parameters to graves' choices
NUM_LAYERS = Constants.NUM_LAYERS
NUM_HIDDEN = Constants.NUM_HIDDEN
BATCH_SIZE = Constants.BATCH_SIZE
NUM_EPOCHS = Constants.NUM_EPOCHS
VALIDATION_SIZE = Constants.VALIDATION_SIZE
ID_STRING = Constants.ID_STRING
OPTIMIZER_DESCR = Constants.OPTIMIZER_DESCR

def create_model(dataset):
    # normalize the dataset
    dataset.normalize()

    # get information about the training set
    max_time_length = max([t for (t, f) in [x.shape for x in dataset.X_train]])
    num_features = dataset.X_train[0].shape[1]
    num_classes = max(TIMITDataset.phoneme_dict.values()) + 2

    # get a validation set
    randomizer = np.arange(len(dataset.X_train))
    np.random.shuffle(randomizer)
    dataset.X_train = dataset.X_train[randomizer]
    dataset.train_timesteps = np.array(dataset.train_timesteps)[randomizer].tolist()
    dataset.X_val = dataset.X_train[:VALIDATION_SIZE]
    dataset.y_val = dataset.y_train[:VALIDATION_SIZE]
    dataset.X_train = dataset.X_train[VALIDATION_SIZE:]
    dataset.y_train = dataset.y_train[VALIDATION_SIZE:]
    dataset.val_timesteps = dataset.train_timesteps[:VALIDATION_SIZE]
    dataset.train_timesteps = dataset.train_timesteps[VALIDATION_SIZE:]
    num_examples = len(dataset.X_train)

    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, None, num_features), name='input')
        targets = tf.sparse_placeholder(tf.int32, name='target')
        seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')


        lstm_cell_forward_list = []
        lstm_cell_backward_list = []
        for i in range(0, NUM_LAYERS):
            lstm_cell_forward_list.append(LSTMCell(NUM_HIDDEN))
            lstm_cell_backward_list.append(LSTMCell(NUM_HIDDEN))

        outputs, f_state, b_state = stack_bidirectional_dynamic_rnn(lstm_cell_forward_list, lstm_cell_backward_list,
                                        inputs, dtype=tf.float32, sequence_length=seq_length)
        
        # prepare the last fully-connected layer, which weights are shared throughout the time steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
        W = tf.Variable(tf.truncated_normal([NUM_HIDDEN,
                                             num_classes],
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        fc_out = tf.matmul(outputs, W) + b
        fc_out = tf.reshape(fc_out, [BATCH_SIZE, -1, num_classes]) # Reshaping back to the original shape
        
        # time major
        fc_out = tf.transpose(fc_out, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, fc_out, seq_length, ignore_longer_outputs_than_inputs=True)
        cost = tf.reduce_mean(loss)
        
        if OPTIMIZER_DESCR == 'adam':
            optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        else:
            optimizer = tf.train.MomentumOptimizer(0.0001,
                                                   0.9).minimize(cost)

        #decoded, log_prob = tf.nn.ctc_greedy_decoder(fc_out, seq_length)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(fc_out, seq_length)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
        
    with tf.Session(graph=graph) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()

        # init saver (has to be after variable initialization)
        saver = tf.train.Saver()
        saver.save(session, os.path.join(Constants.TRAINED_MODELS_FOLDER, ID_STRING + "_initial.ckpt"))

        for curr_epoch in range(NUM_EPOCHS):
            train_cost = train_ler = 0
            start = time.time()

            num_batches_per_epoch = int(num_examples/BATCH_SIZE)
            available_batch_indexes = list(range(0, num_batches_per_epoch))
            for batch in range(num_batches_per_epoch):
                random_batch_index = get_next_batch_index(available_batch_indexes)
                available_batch_indexes.remove(random_batch_index)

                start_index = random_batch_index * BATCH_SIZE
                end_index = (random_batch_index + 1) * BATCH_SIZE 

                # get the data needed for the feed_dict this batch
                batch_seq_length = [time_length for time_length in dataset.train_timesteps[start_index:end_index]]
                batch_inputs = dataset.X_train[start_index:end_index]
                batch_targets = dataset.y_train[start_index:end_index]

                # pad both inputs and targets to max time length in the batch
                batch_inputs = TIMITDataset.pad_train_data(batch_inputs)
                batch_targets = pad_np_arrays(batch_targets)
                batch_dense_shape = np.array([x for x in np.array(batch_targets).shape])

                # get a sparse representation of the targets (tf.nn.ctc_loss needs it for some reason)
                batch_indices, batch_values = array_to_sparse_tuple(np.array(batch_targets))
                
                # run the session on the training data
                feed = {inputs: np.array(batch_inputs),
                        targets: (np.array(batch_indices), np.array(batch_values), batch_dense_shape),
                        seq_length: batch_seq_length}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*BATCH_SIZE
                train_ler += session.run(ler, feed_dict=feed)*BATCH_SIZE

                
                if batch % 1000 == 0:
                    log = "Time: {:.3f}: Batch {:.0f}"
                    logging.info(log.format(time.time() - start, batch))

            train_cost /= num_examples
            train_ler /= num_examples

            # get information on the validation set accuracy
            val_seq_length = dataset.val_timesteps
            val_inputs = dataset.X_val
            val_targets = dataset.y_val

            # pad both inputs and targets to max time length in the batch
            val_inputs = TIMITDataset.pad_train_data(val_inputs)
            val_targets = pad_np_arrays(val_targets)
            val_dense_shape = np.array([x for x in np.array(val_targets).shape])

            # get a sparse representation of the targets (tf.nn.ctc_loss needs it for some reason)
            val_indices, val_values = array_to_sparse_tuple(np.array(val_targets))
            
            # run the session on the training data
            feed = {inputs: np.array(val_inputs),
                    targets: (np.array(val_indices), np.array(val_values), val_dense_shape),
                    seq_length: val_seq_length}

            val_cost = session.run(cost, feed_dict=feed)
            val_ler = session.run(ler, feed_dict=feed)*(VALIDATION_SIZE)


            log = "Epoch {:.0f}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(curr_epoch+1, train_cost, train_ler, val_cost, val_ler,
                             time.time() - start))

        
            # decode a few examples each epoch to monitor progress
            # prepare data and targets
            start_index = 0
            end_index = BATCH_SIZE

            # get the data needed for the feed_dict this batch
            batch_seq_length = [time_length for time_length in dataset.train_timesteps[start_index:end_index]]
            batch_inputs = dataset.X_train[start_index:end_index]
            batch_targets = dataset.y_train[start_index:end_index]
            batch_dense_shape = np.array([x for x in np.array(batch_targets).shape])

            # pad both inputs and targets to max time length in the batch
            batch_inputs = TIMITDataset.pad_train_data(batch_inputs)
            batch_targets = pad_np_arrays(batch_targets)
            
            # get a sparse representation of the targets (tf.nn.ctc_loss needs it for some reason)
            batch_indices, batch_values = array_to_sparse_tuple(np.array(batch_targets))
            
            feed = {inputs: np.array(batch_inputs),
                    targets: (np.array(batch_indices), np.array(batch_values), batch_dense_shape),
                    seq_length: batch_seq_length}

            d = session.run(decoded[0], feed_dict=feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

            for i, seq in list(enumerate(dense_decoded))[:2]:
                seq = [s for s in seq if s != -1]
                inverse_dict = {Constants.TIMIT_PHONEME_DICT[k] : k for k in Constants.TIMIT_PHONEME_DICT}
                original_phoneme_transcription = ' '.join([inverse_dict[k] for k in batch_targets[i]])
                estimated_phoneme_transcription = ' '.join([inverse_dict[k] for k in seq])
                logging.info('Sequence %d' %i)
                logging.info('Original \n%s' %original_phoneme_transcription)
                logging.info('Estimated \n%s' %estimated_phoneme_transcription)

            if curr_epoch % 5 == 0:
                saver.save(session, os.path.join(Constants.TRAINED_MODELS_FOLDER, ID_STRING + "_" + str(curr_epoch) + "e.ckpt"))

        saver.save(session, os.path.join(Constants.TRAINED_MODELS_FOLDER, ID_STRING + "_final.ckpt"))
    return 

def create_model_keras():
    x = Input(shape=(NUM_FEATURES, None, None))
    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(x)
    #for i in range(0, NUM_LAYERS-2):
    #    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(y_pred)
    #y_pred = Bidirectional(LSTM(NUM_HIDDEN), merge_mode='sum')(y_pred)
    model = Model(inputs=x,outputs=y_pred)
    model.compile(loss=ctc_loss, optimizer='adam', metrics=['acc'])
    model.summary()
    return model

if __name__ == "__main__":
    if LOAD_PICKLE == False:
        # create the dataset
        MyTimitDataset = TIMITDataset()
        MyTimitDataset.load()
        MyTimitDataset.to_file()
    else:
        # just load it from pickle
        filename = glob.glob("timit*.pickle")[0]
        with open(filename, "rb") as dataset_file:
            MyTimitDataset = pickle.load(dataset_file)
    create_model(MyTimitDataset)
    evaluate_model()
