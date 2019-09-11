import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.metrics import mean_squared_error
from net_model import RNNModel
from data.read_ext.handle_data import *
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import logging


def plot_learning_curves(X_data, y_data, rnn_model):
    splits = 5
    numpoints = 10
    points = np.linspace(1.0 / numpoints, 1.0, numpoints)
    kval = KFold(n_splits=splits)
    train = []
    cval = []

    for train_index, test_index in kval.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        for point in points:
            split_pt = int(round(point * X_train.shape[0], 0))
            X_tr = X_train[:split_pt]
            y_tr = y_train[:split_pt]

            rnn_model.train_model(X_tr, y_tr)
            y_pred_train = rnn_model.pred_rnn(X_tr)
            y_pred = rnn_model.pred_rnn(X_test)
            train.append(mean_squared_error(y_tr[:, -1, 0], y_pred_train[:, -1, 0]))
            cval.append(mean_squared_error(y_test[:, -1, 0], y_pred[:, -1, 0]))
        break

    plt.plot(points, train, 'b', points, cval, 'r')
    plt.show()


def predict(X_data, y_data, X_score, rnn_model, scalers, col_predict):
    rnn_model.train_model(X_data, y_data)
    y_pred = rnn_model.pred_rnn(X_score)
    pred = y_pred[:, -1, 0]
    return inverse_scale(np.array(pred), scalers, col_predict)


def mae(X_data, y_data, rnn_model):
    folds = KFold(n_splits=5)
    errors = []
    for train_index, test_index in folds.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        rnn_model.train(X_train, y_train)
        y_pred = rnn_model.pred_rnn(X_test)

        error = y_test[:, -1, 0] - y_pred[:, -1, 0]
        errors.extend(error.tolist())

    return mean(map(abs, errors))


def plot_example_tr(X_val, X_data, y_data):
    y_plot = X_val[:, :-1, :]
    pred = predict(X_data, y_data, y_plot)

    for i in range(20):
        ex = random.randint(0, X_val.shape[0] - 1)
        pred_val = pred[ex:ex + 1]
        y_plot = X_val[ex, :-1, 0]
        real = X_val[ex, -1, 0]

        y_plot = inverse_scale(y_plot)
        real = inverse_scale(np.array(real))

        x1 = np.concatenate([np.arange(0, len(y_plot)), np.array([len(y_plot)])])
        x2 = np.array([len(y_plot)])
        y1 = np.concatenate([y_plot, real]).flatten()
        y2 = pred_val

        plt.plot(x1, y1)
        print(y1)
        plt.plot(x2, y2, marker='o', color='r')
        plt.show()


def plot_example_te(X_data, y_data, rnn_model):
    splits = 5
    kval = KFold(n_splits=splits)

    for train_index, test_index in kval.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        rnn_model.train_model(X_train, y_train)
        y_pred = rnn_model.pred_rnn(X_test)
        y_pred = y_pred[:, -1, 0]
        y_pred = inverse_scale(np.array(y_pred))

        for i in range(20):
            ex = random.randint(0, X_test.shape[0] - 1)
            pred_val = y_pred[ex:ex + 1]

            y_plot = X_test[ex, :, 0]
            real = y_test[ex, -1, 0]

            y_plot = inverse_scale(y_plot)
            real = inverse_scale(np.array(real))

            x1 = np.concatenate([np.arange(0, len(y_plot)), np.array([len(y_plot)])])
            x2 = np.array([len(y_plot)])
            y1 = np.concatenate([y_plot, real]).flatten()
            y2 = pred_val

            plt.plot(x1, y1)
            print(y1)
            plt.plot(x2, y2, marker='o', color='r')
            plt.show()


def get_pred_distribution(X_data, y_data, X_score, rnn_model, pred_interval, scalers, col_predict, cv_loops, no_pred,
                          no_samples):
    """
    Creates a probability distribution from the data, using "pred interval" trading days.

    :param X_data:
    :param y_data:
    :param X_score:
    :param rnn_model:
    :param pred_interval:
    :param scalers:
    :param col_predict:
    :param cv_loops:
    :param no_pred:
    :param no_samples:
    :return:
    """
    logging.info("Calculating errors: ")
    errors = []

    kf = KFold(n_splits=5)
    kf.get_n_splits(X_data, y_data)


    for i in range(cv_loops):
        logging.info("Loop " + str(i + 1) + " of " + str(cv_loops))
        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            rnn_model.train_model(X_train, y_train)
            y_pred = rnn_model.pred_rnn(X_test)
            error = y_pred[:, -1, 0] - y_test[:, -1, 0]
            errors.extend(error.tolist())

    logging.info("MAE: " + str(mean(map(abs, errors))))

    preds = []
    logging.info("Calculating predictions")

    for i in range(no_pred):
        logging.info("Prediction " + str(i + 1) + " of " + str(no_pred))
        rnn_model.train_model(X_data, y_data)
        y_pred = rnn_model.pred_rnn(X_score)

        pred = y_pred[0, -1, 0]

        for j in range(no_samples):
            a = pred
            b = random.sample(errors, 1)[0]
            preda = a - b
            preds.append(preda)

    preds = scalers[col_predict].inverse_transform(np.array(preds).reshape(-1, 1))

    # Assumes column was log- transformed
    preds = np.exp(preds)
    now = datetime.now().strftime('%Y%m%d')
    np.savetxt("outputs/preds_" + col_predict + "_" + str(pred_interval) + "_" + str(now) + ".csv", preds, delimiter=",")

    return preds



def get_pred_distribution2(X_data, y_data, X_score, rnn_model, pred_interval, scalers, col_predict):
    logging.info("Calculating errors: ")

    preds = []
    logging.info("Calculating predictions")
    for i in range(0, 150):
        logging.info("Prediction " + str(i) + " of 150")
        rnn_model.train_model(X_data, y_data)
        y_pred = rnn_model.pred_rnn(X_score)

        pred = y_pred[0, -1, 0]
        preds.append(pred)

    preds = scalers[col_predict].inverse_transform(np.array(preds).reshape(-1, 1))
    preds = np.exp(preds)
    np.savetxt("preds_" + col_predict + "_" + str(pred_interval) + ".csv", preds, delimiter=",")

def cross_validate2(data, cols, logcols, pred_interval, n_steps, n_epochs, keep=0.5, l2_reg=0.0):
    train, test = split_train_test(data, 0.5)
    d_train, X_train, y_train, _, scalers = create_ts(train, cols, logcols, pred_interval, n_steps)
    d_test, X_test, y_test, _, scalers = create_ts(test, cols, logcols, pred_interval, n_steps)

    no_cols = len(cols)

    rnn_model = RNNModel(n_epochs=n_epochs)
    rnn_model.define_net(no_cols, n_steps, keep=keep, lambda_l2_reg=l2_reg)
    rnn_model.train_model(X_train, y_train)

    yp_train = rnn_model.pred_rnn(X_train)
    yp_test = rnn_model.pred_rnn(X_test)

    train_mse = mean_squared_error(y_train[:, -1, 0], yp_train[:, -1, 0])
    test_mse = mean_squared_error(y_test[:, -1, 0], yp_test[:, -1, 0])

    return train_mse, test_mse


def min_cross_validate2(data, cols, logcols, pred_interval, n_steps, n_epochs, flip, test_size, keeps, l2_regs):
    train, test = split_train_test(data, test_size)

    if not flip:
        d_train, X_train, y_train, _, scalers = create_ts(train, cols, logcols, pred_interval, n_steps)
        d_test, X_test, y_test, _, scalers = create_ts(test, cols, logcols, pred_interval, n_steps)
    else:
        d_train, X_train, y_train, _, scalers = create_ts(test, cols, logcols, pred_interval, n_steps)
        d_test, X_test, y_test, _, scalers = create_ts(train, cols, logcols, pred_interval, n_steps)

    no_cols = len(cols)

    mses = []
    for keep in keeps:
        for l2_reg in l2_regs:
            rnn_model = RNNModel(n_epochs=n_epochs)
            rnn_model.define_net(no_cols, n_steps, keep=keep, lambda_l2_reg=l2_reg)
            rnn_model.train_model(X_train, y_train)

            yp_train = rnn_model.pred_rnn(X_train)
            yp_test = rnn_model.pred_rnn(X_test)

            train_mse = mean_squared_error(y_train[:, -1, 0], yp_train[:, -1, 0])
            test_mse = mean_squared_error(y_test[:, -1, 0], yp_test[:, -1, 0])

            logging.info("Test MSE for keep:" + str(keep) + ", l2_reg: " + str(l2_reg) + ": " + str(test_mse))

            mses.append((keep, l2_reg, train_mse, test_mse))

    mses.sort(key=lambda x: x[3])

    return mses


def min_cross_validate3(data ,column_predict, cols, logcols, pred_interval, n_steps, n_epochs, test_size, keeps, l2_regs):

    d_train, X_data, y_data, _, scalers = create_ts(data, column_predict, pred_interval, n_steps, True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)

    no_cols = len(cols)

    mses = []
    for keep in keeps:
        for l2_reg in l2_regs:
            rnn_model = RNNModel(n_epochs=n_epochs,column_predict=column_predict)
            rnn_model.define_net(no_cols, n_steps, keep=keep, lambda_l2_reg=l2_reg)
            rnn_model.train_model(X_train, y_train)

            yp_train = rnn_model.pred_rnn(X_train)
            yp_test = rnn_model.pred_rnn(X_test)

            train_mse = mean_squared_error(y_train[:, -1, 0], yp_train[:, -1, 0])
            test_mse = mean_squared_error(y_test[:, -1, 0], yp_test[:, -1, 0])

            logging.info("Test MSE for keep:" + str(keep) + ", l2_reg: " + str(l2_reg) + ": " + str(test_mse))

            mses.append((keep, l2_reg, train_mse, test_mse))

    mses.sort(key=lambda x: x[3])

    return mses

def generate_preds(periods, col_predict, n_steps, n_epochs, store_db=True):
    cv_loops = 5
    no_preds = 200
    no_samples = 1000

    for col in col_predict:
        log = 'outputs/log/series_' + col + '.log'
        logging.basicConfig(filename=log, level=logging.DEBUG)

        data = get_data(col)
        no_cols = len(data.columns)

        for pred_interval in periods:

            logging.info("Predicting for " + col + " pred_interval: " + str(pred_interval))

            data_use = data.copy()
            _, X_data, y_data, X_score, scalers = create_ts(data_use, col, pred_interval, n_steps)

            rnn_model = RNNModel(n_epochs=n_epochs, column_predict=col)
            rnn_model.define_net(no_cols, n_steps, keep=1.0, lambda_l2_reg=0.00001)

            preds = get_pred_distribution(X_data, y_data, X_score, rnn_model, pred_interval, scalers, col, cv_loops, no_preds,
                                  no_samples)

            if store_db:
                pass


def main():
    pred_interval = [15, 25, 50, 100, 150, 200, 250]
    col_predict = ['SPX']
    n_steps = 20
    n_epochs = 1500

    generate_preds(pred_interval, col_predict, n_steps, n_epochs)


def old_main():
    # CONSTS
    # Data transform.
    pred_interval = 50
    n_steps = 20

    # Net definition
    DTYPE = tf.float32

    col_predict = 'SPX'

    # Trainings consts
    n_epochs = 1500


    data = get_data(col_predict)
    no_cols = len(data.columns)

    """
    mses = min_cross_validate3(data, col_predict, cols, logcols, pred_interval, n_steps, 1500, 0.1, keeps=(1.0, ),
                               l2_regs=(0.00001, 0.0001, 0.001, 0.01))

    for (keep, l2_reg, train_mse, test_mse) in mses:
        logging.info('Keep: {:8.5f}, l2_reg: {:8.5f}, train_mse: {:8.5f}, test_mse: {:8.5f}'.format(keep, l2_reg, train_mse, test_mse))
    """

    data, X_data, y_data, X_score, scalers = create_ts(data, col_predict, pred_interval, n_steps)
    rnn_model = RNNModel(n_epochs=n_epochs, column_predict=col_predict)
    # ETH: 1.0, 0.0001
    # SPX: 1.0, 0.00001 (Should test with less reg.)
    # USDMXN: 1.0, 0.0001 (Should test with less reg.)
    # IPC: 1.0, 0.0001
    rnn_model.define_net(no_cols, n_steps, keep=1.0, lambda_l2_reg=0.00001)

    # plot_learning_curves(X_data, y_data, rnn_model)
    # for i in range(100):
    #    print(predict(X_data, y_data, X_score, rnn_model, scalers, col_predict))
    # plot_example_te(X_val)
    # plot_example_tr(X_val)

    # print(mae(X_data, y_data))
    get_pred_distribution(X_data, y_data, X_score, rnn_model, pred_interval, scalers, col_predict)

if __name__ == '__main__':
    main()
