from math import sqrt
from numpy import concatenate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers
import os
import json
import datetime
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
import getopt
import core.model_config as model_config


class Seq2seqConfig():
    GPU = "1"
    n_in = model_config.model_setting['n_in']
    n_out = model_config.model_setting['n_out']
    lstm_encode_size = 200
    lstm_decode_size = 200
    full_size = 100
    test_ratio = 0.1
    model_name = model_config.csv['raw_name']
    experimental_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    csv_file = model_config.csv['raw_file']
    model_in_columns = model_config.model_setting['model_in_columns']
    model_out_columns = model_config.model_setting['model_out_columns']
    n_in_features = len(model_in_columns)
    n_out_features = len(model_out_columns)
    results_dir = "results"

    batch_size = 64
    epochs = model_config.model_setting['epochs']

    graph_name = "seq2seq_keras_%s_lstm_en%d_lstm_de%d_nin%d_nout%d_batch%d_epoch%d_time%s" % (model_name, lstm_encode_size, lstm_decode_size, n_in, n_out, batch_size, epochs, experimental_time)

    def __init__(self):
        self.parse_args()
        self.graph_name = "seq2seq_keras_%s_lstm_en%d_lstm_de%d_nin%d_nout%d_batch%d_epoch%d_time%s" % (self.model_name, self.lstm_encode_size, self.lstm_decode_size, self.n_in, self.n_out, self.batch_size, self.epochs, self.experimental_time)

    #@todo edit parse args
    def parse_args(self):
        try:
            options, args = getopt.getopt(sys.argv[1:], 'cge:', ['common', 'sg', 'epochs='])
            for opt, value in options:
                if opt in ('-c', '--common'):
                    self.model_name = model_config.csv['raw_name']
                    self.csv_file = model_config.csv['raw_file']
                if opt in ('-g', '--sg'):
                    self.model_name = model_config.csv['process_name']
                    self.csv_file = model_config.csv['process_file']
                if opt in ('-e', '--epochs'):
                    self.epochs = int(value)
        except getopt.GetoptError as msg:
            print(msg)
            sys.exit(2)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_dataset(csv_file, model_columns, n_in=30, n_out=1):
    dataset = pd.read_csv(csv_file, header=0, index_col=0)
    columns = list(dataset.columns.values)
    for column in columns:
        if column not in model_columns:
            dataset = dataset.drop(columns=column)
    values = dataset.values
    # integer encode direction
    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    # insert true data
    # reframed = series_to_supervised(values, n_in, n_out)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    print("reframed shape", reframed.shape)
    return reframed, scaler


# split into train and test sets
def split_sets(reframed, n_in, n_out, n_in_features, n_out_features, test_ratio):
    values = reframed.values
    train_size = int(len(values) * (1.0 - test_ratio))
    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs
    n_in_obs = n_in * n_in_features
    n_out_obs = n_out * n_out_features

    train_X, train_y = train[:, :n_in_obs], train[:, n_in_obs:n_in_obs + n_out_obs]
    test_X, test_y = test[:, :n_in_obs], test[:, n_in_obs:n_in_obs + n_out_obs]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_in_features))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_in_features))

    # reshape output to be 2D [samples, timesteps]
    train_y = train_y.reshape((train_y.shape[0], n_out, n_out_features))
    test_y = test_y.reshape((test_y.shape[0], n_out, n_out_features))
    print("train_X shape:", train_X.shape, "train_y shape:", train_y.shape, "test_X shape:", test_X.shape, "test_y shape:", test_y.shape)
    return train_X, train_y, test_X, test_y


def build_model(train_X, train_y, test_X, test_y, lstm_encode_size=200, lstm_decode_size=200, full_size=100, epochs=50, batch_size=72, GPU="0"):
    n_outputs = train_y.shape[1]
    model = Sequential()
    model.add(LSTM(lstm_encode_size, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(lstm_decode_size, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(full_size, activation='relu')))
    model.add(TimeDistributed(Dense(train_y.shape[2])))
    # optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # todo edit loss
    model.compile(loss='mae', optimizer='adam')

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    print("train loss")
    print(history.history['loss'])
    print("test loss")
    print(history.history['val_loss'])
    train_loss = history.history['loss']

    return model, train_loss


def prediction(model, test_X, test_y, n_out, scaler):
    yhat = model.predict(test_X)
    print("yhat.shape:", yhat.shape, "test_y.shape:", test_y.shape)

    test_y = test_y.reshape((test_y.shape[0], n_out))
    yhat = yhat.reshape((yhat.shape[0], n_out))
    # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    # inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(test_y)
    print("inv_yhat.shape:", inv_yhat.shape, "inv_test_y.shape:", inv_y.shape)

    return inv_y, inv_yhat


def evaluate_forecasts(obs, predictions, out_steps):
    # todo edit rmse to mae
    total_rmse = sqrt(mean_squared_error(obs, predictions))
    steps_rmse = []

    for j in range(out_steps):
        temp_dict = {'obs': [], 'predictions': []}
        for i in range(len(obs)):
            temp_dict['obs'].append(obs[i][j])
            temp_dict['predictions'].append(predictions[i][j])
        steps_rmse.append(sqrt(mean_squared_error(temp_dict['obs'], temp_dict['predictions'])))

    return total_rmse, steps_rmse


if __name__ == '__main__':
    config = Seq2seqConfig()
    print("Default configuration:", config.graph_name)
    reframed, scaler = load_dataset(config.csv_file, config.model_in_columns, config.n_in, config.n_out)
    train_X, train_y, test_X, test_y = split_sets(reframed, config.n_in, config.n_out, config.n_in_features, config.n_out_features, config.test_ratio)
    model, train_loss = build_model(train_X, train_y, test_X, test_y, lstm_encode_size=config.lstm_encode_size, lstm_decode_size=config.lstm_encode_size, full_size=config.full_size, epochs=config.epochs, batch_size=config.batch_size, GPU=config.GPU)
    inv_y, inv_yhat = prediction(model, test_X, test_y, config.n_out, scaler)

    total_rmse, steps_rmse=evaluate_forecasts(inv_y, inv_yhat, config.n_out)
    print('Test RMSE: %.3f' % total_rmse)
    print('Test Step RMSE: ', steps_rmse)

    # result
    results = {"test": inv_y.tolist(), "prediction": inv_yhat.tolist(), "train_loss": train_loss, "rmse": total_rmse, "steps_rmse": steps_rmse}
    if not os.path.exists(config.results_dir):
        os.mkdir(config.results_dir)
    with open("results/{}.json".format(config.graph_name), 'w') as fout:
        fout.write(json.dumps(results))

