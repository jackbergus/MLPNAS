import os
import pickle
import warnings
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
import keras.backend as K


# from CONSTANTS import *


class MLPSearchSpace(object):
    def __init__(self, target_classes, nodes, act_funcs):
        self.nodes = nodes
        self.target_classes = target_classes
        self.act_funcs = act_funcs
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        # nodes = [8, 16, 32, 64, 128, 256, 512]
        # act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(self.nodes)):
            for j in range(len(self.act_funcs)):
                layer_params.append((self.nodes[i], self.act_funcs[j]))
                layer_id.append(len(self.act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab

    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence


def precision(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0.0:
        return 0.0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


class MLPGenerator(MLPSearchSpace):
    def __init__(self, conf):
        self.target_classes = conf.TARGET_CLASSES
        self.mlp_optimizer = conf.MLP_OPTIMIZER
        self.mlp_lr = conf.MLP_LEARNING_RATE
        self.mlp_decay = conf.MLP_DECAY
        self.mlp_momentum = conf.MLP_MOMENTUM
        self.mlp_dropout = conf.MLP_DROPOUT
        self.mlp_loss_func = conf.MLP_LOSS_FUNCTION
        self.mlp_one_shot = conf.MLP_ONE_SHOT
        self.metrics = []
        for x in conf.METRICS:
            if x == "precision":
                self.metrics.append(precision)
            elif x == "recall":
                self.metrics.append(recall)
            elif x == "f1score":
                self.metrics.append(f1_score)
            else:
                self.metrics.append(x)
        super().__init__(conf.TARGET_CLASSES, conf.nodes, conf.activation_functions)
        if self.mlp_one_shot:
            self.weights_file = 'LOGS/shared_weights.pkl'
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)

    def load_from_configuration_folder(self, folder_name):
        self.weights_file = os.path.join(folder_name, 'shared_weights.pkl')
        with open(self.weights_file, 'rb') as f:
            self.shared_weights = pickle.load(f)
        return self.shared_weights

    def create_model(self, sequence, mlp_input_shape):
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()
        if len(mlp_input_shape) > 1:
            model.add(Flatten(name='flatten', input_shape=mlp_input_shape))
            for i, layer_conf in enumerate(layer_configs):
                if layer_conf == 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        else:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))
                elif layer_conf == 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        return model

    def compile_model(self, model):
        if self.mlp_optimizer == 'sgd':
            optim = optimizers.SGD(lr=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum)
        else:
            optim = getattr(optimizers, self.mlp_optimizer)(lr=self.mlp_lr, decay=self.mlp_decay)
        model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
        return model

    def update_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                      'weights': layer.get_weights()},
                                                                     ignore_index=True)
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)

    def set_model_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                j += 1

    def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
            self.update_weights(model)  # Serialises the weights to the file
        else:
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
        return history
