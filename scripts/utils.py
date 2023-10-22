import numpy as np

def find_nearest(list_l, value):
    """Finds an index of the nearest element of a list

    Parameters
    ----------
    list_l : list
        a list of numeric elements
    value : float or int
        value for finding the nearest element in a list

    Returns
    -------
    idx : int
        index of the nearest element of a list
    """
    array = np.asarray(list_l)
    idx = (np.abs(array - value)).argmin()
    return idx

def CE(targets,predictions):
  """Computing the coefficient of efficiency (CE)

    Parameters
    ----------
    targets : numpy array
        observed values
    predictions : numpy array
        predicted values

    Returns
    -------
    CE_ : float
        coefficient of efficiency (CE)
  """

  return 1 - np.sum((targets-predictions)**2) / np.sum((targets- np.mean(targets))**2)

def RE(targets,predictions,train_mean):
  """Computing the reduction of error (RE)

    Parameters
    ----------
    targets : numpy array
        observed values
    predictions : numpy array
        predicted values
    train_mean: float
        mean of target values that was used in training

    Returns
    -------
    RE_ : float
        reduction of error (RE)
  """
  return 1 - np.sum((targets-predictions)**2) / np.sum((targets-train_mean)**2)


