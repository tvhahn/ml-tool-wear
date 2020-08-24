import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
import pathlib
import pandas as pd
from pathlib import Path
import h5py
import pickle

# import random

def scaler(x, min_val, max_val, lower_norm_val, upper_norm_val):
    """Scale the signal between a min and max value
    
    Parameters
    ===========
    x : ndarray
        Signal that is being normalized

    max_val : int or float
        Maximum value of the signal or dataset

    min_val : int or float
        Minimum value of the signal or dataset

    lower_norm_val : int or float
        Lower value you want to normalize the data between (e.g. 0)

    upper_norm_val : int or float
        Upper value you want to normalize the data between (e.g. 1)

    Returns
    ===========
    x : ndarray
        Returns a new array that was been scaled between the upper_norm_val
        and lower_norm_val values

    """

    # https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
    col, row = np.shape(x)
    for i in range(col):
        x[i] = np.interp(x[i], (min_val, max_val), (lower_norm_val, upper_norm_val))
    return x


class DataPrep:
    def __init__(self, data_path):

        self.data_file = data_path

        # load the data from the matlab file
        m = sio.loadmat(self.data_file, struct_as_record=True)

        # store the 'mill' data in a seperate np array
        self.data = m["mill"]

        self.field_names = self.data.dtype.names

    def create_labels(self):
        """Function that will create the label dataframe from the mill data set"""

        # store the field names in the data np array in a tuple, l
        l = self.field_names

        # create empty dataframe for the labels
        df_labels = pd.DataFrame()

        # get the labels from the original .mat file and put in dataframe
        for i in range(7):
            # list for storing the label data for each field
            x = []

            # iterate through each of the unique cuts
            for j in range(167):
                x.append(self.data[0, j][i][0][0])
            x = np.array(x)
            df_labels[str(i)] = x

        # add column names to the dataframe
        df_labels.columns = l[0:7]

        # create a column with the unique cut number
        df_labels["cut_no"] = [i for i in range(167)]

        def tool_state(cols):
            """Add the label to the cut. Categories are:
            Healthy Sate (label=0): 0~0.2mm flank wear
            Degredation State (label=1): 0.2~0.7mm flank wear
            Failure State (label=2): >0.7mm flank wear 
            """
            # pass in the tool wear, VB, column
            vb = cols

            if vb < 0.2:
                return 0
            elif vb >= 0.2 and vb < 0.7:
                return 1
            elif pd.isnull(vb):
                pass
            else:
                return 2

        # apply the label to the dataframe
        df_labels["tool_class"] = df_labels["VB"].apply(tool_state)

        return df_labels

    def scale_data(self, lower, upper):

        # get the min-max values for the smcAC and smcDC signals
        def get_min_max(x):

            # flatten the input array http://bit.ly/2MQuXZd
            flat_vector = np.concatenate(x).ravel()

            min_val = min(flat_vector)
            max_val = max(flat_vector)

            return min_val, max_val

        l = self.field_names

        # iterate through each signal type in the data to scale
        for i in l[7:]:
            list_a = []  # empty list
            print(i)

            # iterate through each data-point in the samples
            for j in range(167):
                a = self.data[0, j][i]

                # exclude some samples since they are bad
                if j not in [17, 94]:
                    for k in get_min_max(a):
                        list_a.append(k)
                else:
                    pass

            # get min-max values
            min_val_a = min(list_a)
            max_val_a = max(list_a)
            print(min_val_a, max_val_a, "\n")

            # scale each sample
            for j in range(167):
                a = self.data[0, j][i]
                a = scaler(a, min_val_a, max_val_a, lower, upper)

    def create_tensor(
        self, data_sample, signal_names, start, end, window_size, stride=8
    ):
        """Create a tensor from a cut sample. Final tensor will have shape: 
        [# samples, # sample len, # features/sample]

        Parameters
        ===========
        data_sample : ndarray
            single data sample containing all the signals

        signal_names : tuple
            tuple of all the signals that will be added into the tensor
        """

        s = signal_names[::-1]  # only include the six signals, and reverse order
        c = data_sample[s[0]].reshape((9000, 1))

        for i in range(len(s)):
            try:
                a = data_sample[s[i + 1]].reshape((9000, 1))  # reshape to make sure
                c = np.hstack((a, c))  # horizontal stack
            except:
                # reshape into [# samples, # sample len, # features/sample]
                c = c[start:end]
                c = np.reshape(c, (c.shape[0], -1))

        dummy_array = []
        # fit the strided windows into the dummy_array until the length
        # of the window does not equal the proper length
        for i in range(c.shape[0]):
            windowed_signal = c[i * stride : i * stride + window_size]
            if windowed_signal.shape == (window_size, 6):
                dummy_array.append(windowed_signal)
            else:
                break

        c = np.array(dummy_array)
        # print(c.shape)

        return c

    def return_xy(self, df_labels, data, signal_names, window_size, stride=8, save_pickles=False, track_y=False):

        temp_cuts = []  # temporary list to hold all the windowed cuts
        temp_labels = []
        track_temp_labels = []
        X = []  # instantiate X's
        y = []  # instantiate y's
        y_track = [] 

        # iterate throught the df
        for i in df_labels.itertuples():
            cut_data_ind = self.create_tensor(
                data[0, i.cut_no],
                signal_names,
                i.window_start,
                i.window_end,
                window_size,
                stride,
            )

            if save_pickles == True:
                filename = 'pickle_saves/{}.pickle'.format(i.cut_no)

                pathlib.Path('./pickle_saves').mkdir(parents=True, exist_ok=True)

                with open(filename, 'wb') as f:
                    pickle.dump(cut_data_ind, f)


            temp_cuts.append(cut_data_ind)
            temp_labels.append(i.tool_class)
            track_temp_labels.append([i.tool_class, i.cut_no, i.case])

        for i, tool_class in enumerate(temp_labels):
            for cut_split in temp_cuts[i]:
                y.append(tool_class)
                X.append(cut_split)

        for i, tool_class in enumerate(track_temp_labels):
            for j, cut_split in enumerate(temp_cuts[i]):
                # tool_class[1] = str(tool_class[1]).join(['_',str(j)])
                # tool_class[1] = tool_class[1]+j/10000
                # print(tool_class[1])
                y_track.append([tool_class[0],tool_class[1]+j/10000, tool_class[2]])

        # vertical stack the X list (make it into an array)
        X = np.array(X)
        # print("Shape of X:", X.shape)
        y = np.array(y)
        # print("Shape of y:", y.shape)
        y_track = np.array(y_track)

        if track_y == True:
            df_y = pd.DataFrame(y_track, columns=['class', 'counter', 'case'])
            return X, y, df_y
        else:
            return X, y

    def remove_classes(self, class_to_remove, y_val_slim, X_val_slim):
        """Funciton to remove classes from train/val set"""

        # start with y_valid_slim
        index_to_delete = []
        for i, class_digit in enumerate(y_val_slim):
            if class_digit in class_to_remove:
                index_to_delete.append(i)

        y_val_slim = np.delete(y_val_slim, index_to_delete)
        X_val_slim = np.delete(X_val_slim, index_to_delete, axis=0)

        return X_val_slim, y_val_slim

    def train_test_split(
        self,
        df_labels,
        train_cut_no=[1, 2, 3],
        val_cut_no=[11, 13],
        test_cut_no=[5, 10, 15],
        window_size=64,
        stride=64,
        class_to_remove=[2],
        return_hdf5=False,
        print_shapes=True,
        save_pickles=False,
    ):

        # create the label dataframes for each of the train, val, test sets
        df_train = df_labels[df_labels["cut_no"].isin(train_cut_no)]
        df_val = df_labels[df_labels["cut_no"].isin(val_cut_no)]
        df_test = df_labels[df_labels["cut_no"].isin(test_cut_no)]

        signal_names = self.field_names[7:]

        self.X_train, self.y_train = self.return_xy(
            df_train, self.data, signal_names, window_size, stride, save_pickles
        )

        self.X_val, self.y_val = self.return_xy(
            df_val, self.data, signal_names, window_size, stride, save_pickles
        )

        self.X_test, self.y_test = self.return_xy(
            df_test, self.data, signal_names, window_size, stride, save_pickles
        )

        self.X_train_slim, self.y_train_slim = self.remove_classes(
            class_to_remove, self.y_train, self.X_train
        )
        self.X_val_slim, self.y_val_slim = self.remove_classes(
            class_to_remove, self.y_val, self.X_val
        )

        if print_shapes == True:

            print("Shape of X_train:", self.X_train.shape)
            print("Shape of y_train:", self.y_train.shape)

            print("Shape of X_val:", self.X_val.shape)
            print("Shape of y_val:", self.y_val.shape)

            print("Shape of X_test:", self.X_test.shape)
            print("Shape of y_test:", self.y_test.shape)

            print("Shape of X_train_slim:", self.X_train_slim.shape)
            print("Shape of y_train_slim:", self.y_train_slim.shape)

            print("Shape of X_val_slim:", self.X_val_slim.shape)
            print("Shape of y_val_slim:", self.y_val_slim.shape)


        if return_hdf5 == True:
            with h5py.File("X_train.hdf5", "w") as f:
                dset = f.create_dataset("X_train", data=self.X_train)
            with h5py.File("y_train.hdf5", "w") as f:
                dset = f.create_dataset("y_train", data=self.y_train)

            with h5py.File("X_train_slim.hdf5", "w") as f:
                dset = f.create_dataset("X_train_slim", data=self.X_train_slim)
            with h5py.File("y_train_slim.hdf5", "w") as f:
                dset = f.create_dataset("y_train_slim", data=self.y_train_slim)

            with h5py.File("X_val.hdf5", "w") as f:
                dset = f.create_dataset("X_val", data=self.X_val)
            with h5py.File("y_val.hdf5", "w") as f:
                dset = f.create_dataset("y_val", data=self.y_val)

            with h5py.File("X_val_slim.hdf5", "w") as f:
                dset = f.create_dataset("X_val_slim", data=self.X_val_slim)
            with h5py.File("y_val_slim.hdf5", "w") as f:
                dset = f.create_dataset("y_val_slim", data=self.y_val_slim)

            with h5py.File("X_test.hdf5", "w") as f:
                dset = f.create_dataset("X_test", data=self.X_test)
            with h5py.File("y_test.hdf5", "w") as f:
                dset = f.create_dataset("y_test", data=self.y_test)

        return (
            self.X_train,
            self.y_train,
            self.X_train_slim,
            self.y_train_slim,
            self.X_val,
            self.y_val,
            self.X_val_slim,
            self.y_val_slim,
            self.X_test,
            self.y_test,
        )
