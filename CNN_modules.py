from pythtb import *
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def get_model(delta=0.2, theta=np.pi / 2, d=0):
    """returns a pythtb formated model of Halden

    Keyword Arguments:
        delta {float} -- onsite difference (default: {0.2})
        theta {float} -- angle (default: {np.pi/2})
        d {float} -- onsite shift (default: {0})

    Returns:
        pythtb_model -- the model
    """
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)

    t = -1.0
    t2 = 0.1 * np.exp((1.0j) * theta)
    t2c = t2.conjugate()

    my_model.set_onsite([-delta + d, delta + d])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    my_model.set_hop(t2, 0, 0, [1, 0])
    my_model.set_hop(t2, 1, 1, [1, -1])
    my_model.set_hop(t2, 1, 1, [0, 1])
    my_model.set_hop(t2c, 1, 1, [1, 0])
    my_model.set_hop(t2c, 0, 0, [1, -1])
    my_model.set_hop(t2c, 0, 0, [0, 1])
    return my_model


def get_wav(my_model, nk=10):
    """returns the wavefunction on a uniform mesh

    Arguments:
        my_model {pythtb model} -- model

    Keyword Arguments:
        nk {float} -- num of k points (default: {10})

    Returns:
        wav -- wave function on a nkxnkmesh
    """
    k_vec = my_model.k_uniform_mesh([nk, nk])
    evals, evec = my_model.solve_all(k_vec, eig_vectors=True)
    return evec


def reshape_evec(evec, phase=True):
    """reshapping vector to CNN readable format

    Arguments:
        evec {nparray} -- eigen vectors

    Keyword Arguments:
        phase {bool} -- decides if the space is made up of r(exp(i*phi)) or Re and Im part (default: {True})

    Returns:
        nparray -- formated arrray of size n x nk x nbnd x norb
    """
    nk = int(np.sqrt(evec.shape[1]))
    nbnd = norb = evec.shape[0]
    evec = np.swapaxes(evec, 0, 1)
    evec_new = evec.reshape(nk, nk, nbnd, norb)
    a = []
    b = []
    for i in range(int(nbnd / 2)):
        for j in range(norb):
            if phase:
                a.append(np.angle(evec_new[:, :, i, j]))
                a.append(np.absolute(evec_new[:, :, i, j]))
            else:
                a.append(np.real(evec_new[:, :, i, j]))
                a.append(np.imag(evec_new[:, :, i, j]))

    a = np.array(a).T
    return a


def get_chern(my_model, nk=8, k_origin=0):
    """get the chern number of the model

    Arguments:
        my_model {pythtb model} -- model to vcalculate the chern number

    Keyword Arguments:
        nk {int}         -- nk for calculating the chern number (default: {8})
        k_origin {float} -- starting point for calculating the k mesh
    Returns:
        int -- chern number
    """
    nkx = nky = nk
    kx = np.linspace(k_origin, k_origin + 1, num=nkx)
    ky = np.linspace(k_origin, k_origin + 1, num=nky)
    my_array_2 = wf_array(my_model, [nkx, nky])
    # solve model at all k-points
    for i in range(nkx):
        for j in range(nky):
            (eval, evec) = my_model.solve_one([kx[i], ky[j]], eig_vectors=True)
            # store wavefunctions
            my_array_2[i, j] = evec
    my_array_2.impose_pbc(0, 0)
    my_array_2.impose_pbc(1, 1)
    return int(np.round(my_array_2.berry_flux([0]) / (2 * np.pi)))


# --- Create data for NN


def process_y(Y):
    """pre process the output Y
    example [0,1,2]-[[1,0,0],[0,1,0],[0,0,1]]

    Arguments:
        Y {list} -- chern numbers

    Returns:
        ndarray -- catorgies of values
    """
    from sklearn import preprocessing

    enc = preprocessing.OneHotEncoder()
    enc.fit(np.array(Y).reshape(-1, 1))
    return enc, enc.transform(np.array(Y).reshape(-1, 1)).toarray()


def get_NNmodel(X_train,model_type="simple_cnn"):
    """return the NN model
    Arguments:
        X_train {ndarray} -- to calculate input dims
    Keyword Arguments:
        model_type {str} -- Type of model
        right now supports simple_cnn (default: {"simple_cnn"})

    Returns:
        keras.model -- keras model
    """
    dim=X_train.shape[1:]
    model = Sequential()
    model.add(
        Conv2D(
            10,
            kernel_size=2,
            activation="relu",
            input_shape=(dim[0],dim[1],dim[2]),
        )
    )
    model.add(Dropout(0.4))
    model.add(Conv2D(5, kernel_size=2, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
