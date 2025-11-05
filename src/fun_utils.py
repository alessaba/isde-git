from pandas import read_csv
import numpy as np

def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    #Nico did this
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    
    n_images = x.shape[0]

    n_tr = int(tr_fraction*n_images)
    n_ts = n_images - n_tr

    idx = np.linspace(0, n_images, num=n_images, endpoint=False, dtype='int')
    np.random.shuffle(idx)  

    tr_idx = idx[:n_tr]   
    ts_idx = idx[n_tr:] 

    xtr = x[tr_idx,:] 
    ytr = y[tr_idx]
    xts = x[ts_idx,:] 
    yts = y[ts_idx]
    return xtr, ytr, xts, yts

