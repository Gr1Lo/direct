import numpy as np
import scipy as sp

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

def direct_statistics(av_df, train_mean, clim_name='avg summer temperature'):
    """Prints calibration and verification statistics

      Parameters
      ----------
      av_df : pandas dataframe object
          dataframe with predictions that averaged by the year
      train_mean: float
          mean of target values that was used in training
      clim_name: str, optional:
          name of column with observed values, default is
          'avg summer temperature'
    """
    #Calibration and verification statistics
    print('corr (Spearman): ', np.round(sp.stats.spearmanr(av_df[clim_name],
                                                           av_df['preds'])[0],2))
    print('CE: ', round(CE(av_df[clim_name],av_df['preds']),2))
    print('RE: ', round(RE(av_df[clim_name],av_df['preds'],train_mean),2))
    #Root Mean Square Error
    RMSE = ((av_df[clim_name] - av_df['preds']) ** 2).mean() ** .5
    print('RMSE: ', round(RMSE,2))



