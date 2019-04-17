import tensorflow as tf
#from deepspeech.model import Model
import scipy.io.wavfile as wav
import sys
import glob
import argparse
import json
import re
import glob
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from deepspeech.utils import audioToInputVector
from utils.constants import Constants
from utils.utils import to_csv, fix_seq_length, apply_pca
import scipy.io.wavfile as wav
import pickle
import os

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def create_model():
    # These constants control the beam search decoder

    # Beam width used in the CTC decoder when building candidate transcriptions
    BEAM_WIDTH = 500

    # The alpha hyperparameter of the CTC decoder. Language Model weight
    LM_WEIGHT = 1.75

    # The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
    WORD_COUNT_WEIGHT = 1.00

    # Valid word insertion weight. This is used to lessen the word insertion penalty
    # when the inserted word is part of the vocabulary
    VALID_WORD_COUNT_WEIGHT = 1.00

    # These constants are tied to the shape of the graph used (changing them changes
    # the geometry of the first layer), so make sure you use the same constants that
    # were used during training

    deepspeech_model = Model(args.model, N_FEATURES,
                             N_CONTEXT, args.alphabet, BEAM_WIDTH)

    if args.lm and args.trie:
        deepspeech_model.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                                             WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

    return deepspeech_model


def test_model(deepspeech_model):
    # load transcription data
    transcriptions = json.load(open(args.transcription_json))

    i = 0
    for i in range(len(transcriptions)):
        audio_file = args.audio_folder + '/' + str(i) + '.wav'
        sr, audio = wav.read(audio_file)

        assert sr == 16000, "Sample rate is not 16k! All other sample rates are unsupported as of now."

        audio_length = len(audio) * (1 / sr)

        y = deepspeech_model.stt(audio, sr)
        y_true = transcriptions[str(i)].split(',')[0]

        print('Estimated: {} \n True: {}'.format(y, y_true))

        i += 1

def get_model_output_10_classes(filename):
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tensors_and_ops = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=graph) as sess:
            output_op = graph.get_operation_by_name('Minimum_3').outputs[0]
            xs = []
            ys = []
            filename_list = []
            for idx, filename in enumerate(glob.glob(os.path.join(Constants.AUDIO_DATA_FOLDER, '*wav*', '*.wav'))):
                name = filename.split('/')[-2:]
                if int(name[-1].strip('.wav')) < 1000:
                    continue
                y = name[-1].strip('.wav')
                y = str(int(y) - 1000)
                name = '/'.join(name)
                name = name.replace('.wav', '')
                filename_list.append(name)
                if idx % 50 == 0:
                    print(name)
                fs, audio = wav.read(filename)
                x = audioToInputVector(audio, fs, N_FEATURES, N_CONTEXT)
                out = sess.run(output_op, {'input_node:0': [
                               x], 'input_lengths:0': [len(x)]})
                xs.append(out)
                ys.append(y)
            xs = fix_seq_length(xs, length=20)
            xs = apply_pca(xs, n_components=25)
            xs = np.array([np.ravel(x) for x in xs])
            to_csv(xs, ys, os.path.join(Constants.DATA_FOLDER, 'audio10classes.csv'),
                   filename_list=filename_list)

def get_model_output_100_classes(filename):
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tensors_and_ops = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=graph) as sess:
            output_op = graph.get_operation_by_name('Minimum_3').outputs[0]
            xs = []
            ys = []
            for idx, filename in enumerate(glob.glob(os.path.join(Constants.AUDIO_DATA_FOLDER, '*wav*', '*.wav'))):
                name = filename.split('/')[-2:]
                y = name[-1].strip('.wav')
                print(name[-1])
                name = '/'.join(name)
                name = name.replace('.wav', '')
                if idx % 50 == 0:
                    print(name)
                fs, audio = wav.read(filename)
                x = audioToInputVector(audio, fs, N_FEATURES, N_CONTEXT)
                out = sess.run(output_op, {'input_node:0': [
                               x], 'input_lengths:0': [len(x)]})
                out = np.ravel(out)
                xs.append(out)
                ys.append(y)
            xs = fix_seq_length(xs, length=20)
            xs = apply_pca(xs, n_components=25)
            xs = np.array([np.ravel(x) for x in xs])
            to_csv(xs, ys, os.path.join(Constants.DATA_FOLDER, 'audio10classes.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test script running the deepspeech model on the OS X speaker dataset.')
    parser.add_argument('model', type=str,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('audio_folder', type=str,
                        help='Path to the audio directory containing the files (WAV format)')
    parser.add_argument('transcription_json', type=str,
                        help='Path to the json file containing the transcriptions of the WAV files')
    parser.add_argument('alphabet', type=str,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('lm', type=str, nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('trie', type=str, nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    args = parser.parse_args()

    #deepspeech_model = create_model()
    # test_model(deepspeech_model)
    get_model_output_10_classes(args.model)
