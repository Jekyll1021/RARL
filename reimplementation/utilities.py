import numpy as np
import matplotlib.pyplot as plt
import glob
import torch


def save_model(model, log, name):
    torch.save(model.state_dict(), name)
    np.savetxt(name + '.log', log)
    fig = plt.figure()
    plt.plot(log)
    plt.show()
    fig.savefig(name + '.png')


def get_training_data(human=False):
    files = glob.glob('data/training/demo_*.csv')
    return get_data_from_raw(files, human)
    # files = [
    #         'data/demo_slow.csv',
    #         'data/demo_fast.csv',
    #         'data/demo_inverse.csv',
    #         'data/demo_hesitant.csv',
    #         ]
    # return get_data(files)


def get_training_data_lstm():
    files = glob.glob('data/training/seq_*.csv')
    return get_data_from_raw_lstm(files)


def get_test_data(human=False):
    # files = glob.glob('data/test/demo_*.csv')
    # return get_data_from_raw(files, human)
    files = glob.glob('data/demo_fast.csv')
    return get_data(files, human)
    # files = [
    #     'data/demo_hesitant.csv'
    # ]
    # return get_data_from_raw(files)


def get_data(files, human=False):
    dataset = None
    for fn in files:
        data = np.genfromtxt(fn, delimiter=',')
        # get rid of timestamp and add next state
        data = data[:, 1:]
#        next_state = np.append(data[1:, 0], np.nan).reshape(-1, 1)
#        data = np.hstack((data, next_state))
        if human:
            next_state = data[:, 1] - np.append(data[1:, 1], np.nan)
        else:
            next_state = data[:, 0] - np.append(data[1:, 0], np.nan)
        data = np.hstack((data, next_state.reshape(-1, 1) ))

        # delete rows which contains nan
        data = data[~np.isnan(data).any(axis=1)].astype('float')

        # add to dataset
        if dataset is None:
            dataset = data
        else:
            dataset = np.concatenate((dataset, data))

    return dataset


def get_data_from_raw(files, human=False):
    dataset = None
    for fn in files:
        data = np.genfromtxt(fn, delimiter=',')
        # get rid of gps and axis z
        data = data[:, :-4]
        # compute velocity
        v_c = [np.nan]
        v_h = [np.nan]
        for i in range(1, len(data)):
            if np.isnan(data[i,1]) and np.isnan(data[i-1, 1]):
                v_c.append(np.nan)
            else:
                v_c.append( (data[i-1,1]-data[i, 1]) / (data[i,0] - data[i-1,0]) )

            if np.isnan(data[i,2]) and np.isnan(data[i-1, 2]):
                v_h.append(np.nan)
            else:
                v_h.append( (data[i,2]-data[i-1, 2]) / (data[i,0] - data[i-1,0]) )

        v_c = np.array(v_c).reshape(-1,1)
        v_h = np.array(v_h).reshape(-1,1)
        data = np.hstack((data, v_c))
        data = np.hstack((data, v_h))

        # get rid of timestamp and add next state
        data = data[:, 1:]
        if human:
            next_state = data[:, 1] - np.append(data[1:, 1], np.nan)
        else:
            next_state = data[:, 0] - np.append(data[1:, 0], np.nan)
        data = np.hstack((data, next_state.reshape(-1, 1) ))

        # delete rows which contains nan
        data = data[~np.isnan(data).any(axis=1)].astype('float')

        # add to dataset
        if dataset is None:
            dataset = data
        else:
            dataset = np.concatenate((dataset, data))

    return dataset


def get_data_from_raw_lstm(files):

    dataset = () 

    # Takes data into
    for fn in files:
        # Data format input:
        # Timestamp | v_c | x_h | y_h | next_delta_y

        # Data format output:
        # v_c | x_h | y_h | next_delta_y

        data = np.genfromtxt(fn,  delimiter=',', skip_header=1)

        # replace nan with 0
        np.nan_to_num(data, copy=False)

        # get rid of timestamp and the last data
        dataset += (data[:-1, 1:], )

    return dataset
