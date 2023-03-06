from matplotlib.bezier import NonIntersectingPathException
import os
import sys
import pandas as pd
import numpy as np
import re
import scipy as sp
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import seaborn as sns
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.utils import resample
from tqdm import tqdm
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


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


def rwl2pandas(rwl_path, no_data_value=-9999, ind_length = 8, pth_path=None, 
               vals_name='vals'):
  """Reading rwl-files and inserting values to pandas dataframe

  Parameters
  ----------
  rwl_path : str
      path to rwl-file
  no_data_value : float or int, optional
      a value that will be interpreted as nodata or end of series,
      default is -9999
  ind_length : int, optional
      number of symbols at the beginning of the string that determines the name 
      of a series, default is 8
  pth_path : str, optional
      path to pth-file, which contains data about the age of a tree in the first 
      year of series, which can be one of two types:
      1) Text file with the age of trees at the first year of observation 
      delimited  with newline symbol (\n) in corresponding order to series 
      in rwl-file
      2) Text file with the name of a series in the first column and with the 
      year of the first observation in the second column 
      default is None
  vals_name : str, optional
      name for column of values in output dataframe, default is 'vals'

  Returns
  -------
  df : pandas dataframe object
      Dataframe with columns: 
          'years' - year of the observation, 
          vals_name - measurand, 
          'age' - age of the tree at the observation, 
          'file' - name of the series
  """
  
  f_age = [] #list with first ages
  f_age_files = []
  if pth_path is not None:
        with open(pth_path) as file:
          while (line := file.readline().rstrip()):
            if len(line.split())>1:
                #for pth-file with multiple columns
                f_age.append(int(line.split()[1]))
                f_age_files.append(line.split()[0].strip())
            else:
                #for pth-file with one column
                f_age.append(int(line[0]))
    
  else:
    #in the case when pth-file is not specificated all first observations 
    #in a series are assumed to be zero     
    with open(rwl_path) as file:
      while (line := file.readline().rstrip()):
        f_age.append(0)

  all_l = []
  ind_file_list = []
  df = pd.DataFrame(columns=['years', vals_name, 'age', 'file'])
  cou = 0
  with open(rwl_path) as file:
    while (line := file.readline().rstrip()):
        ind_file = line[:ind_length].strip() #series name
        obs = line[ind_length:].strip().split()
        #year of first observation in the row
        year_line = int(obs[0]) 
        #checking whether it's the first appearance of the series name or not 
        #for an age assignment
        if ind_file not in ind_file_list:
          if not f_age_files:
              age_first = f_age[cou]
          else:
              ind = f_age_files.index(ind_file)
              #for the multiple column type of pth-file first age is defined by 
              #subtraction year of first observation and pith year
              age_first = year_line-f_age[ind]

          ind_file_list.append(ind_file)
          cou += 1

        #appending every observation to the list of lists that further will be 
        #inserted to output df
        for i in range(1, len(obs)):
            ob = float(obs[i])
            if ob != no_data_value:
              all_l.append([year_line, ob, age_first, ind_file])

            year_line += 1
            age_first += 1

  df = pd.DataFrame.from_records(all_l,columns=['years', vals_name, 
                                                'age', 'file'])
  return df


def read_meteo(file_p, sep = '\t', names=list(range(0,12)), months=[5,6,7], 
               clim_name='avg summer temperature'):
  
    """Reading meteo-data table where columns correspond to months and 
    rows correspond to years

    Parameters
    ----------
    file_p : str
        path to meteo-data csv-file
    sep : str, optional
        separator parameter for pandas.read_csv, default is '\t'
    names : list, optional
        list of colnames in meteo-data csv-file, default is [0,1,...,11]
    months : list, optional
        list of colnames for averaging, default is [5,6,7] (summer months)
    clim_name: str, optional:
        name of averaged value in output dataframe, default is 'avg summer 
        temperature'

    Returns
    -------
    df : pandas dataframe object
        Dataframe with columns: 
            'years' - year of the observation, 
            clim_name - averaged values for each year
    """

    df_meteo = pd.read_csv(file_p, sep=sep, names=names)
    #selecting by months list
    df_meteo = df_meteo[months]
    #creating a new column with averaged values
    df_meteo.loc[:,clim_name] = df_meteo.mean(axis=1)
    df_meteo.loc[:,'years'] = df_meteo.index
    return df_meteo[['years', clim_name]]

def direct_read(rwl, meteo, no_data_value=-9999, ind_length=8, pth_path=None,
                clim_name='avg summer temperature', vals_name='vals', 
                meteo_sep = '\t', meteo_names = list(range(0,12)), 
                meteo_months = [5,6,7]):
  
    """Reading and merging rwl-file and meteo-data

    Parameters
    ----------
    rwl : str
        path to rwl-file
    meteo : str
        path to meteo-data csv-file
    no_data_value : float or int, optional
      a value that will be interpreted as nodata or end of series,
      default is -9999
    ind_length : int, optional
        number of symbols at the beginning of the string that determines 
        the name of a series, default is 8
    pth_path : str, optional
        path to pth-file, which contains data about the age of a tree 
        in the first year of series, which can be one of two types:
        1) Text file with the age of trees at the first year of observation 
        delimited  with newline symbol (\n) in corresponding order to series 
        in rwl-file
        2) Text file with the name of a series in the first column and with the 
        year of the first observation in the second column 
        default is None
    vals_name : str, optional
        name for column of values in output dataframe, default is 'vals'
    meteo_sep : str, optional
        separator parameter for pandas.read_csv, default is '\t'
    meteo_names : list, optional
        list of colnames in meteo-data csv-file, default is [0,1,...,11]
    meteo_months : list, optional
        list of colnames for averaging, default is [5,6,7] (summer months)
    clim_name: str, optional:
        name of averaged value in output dataframe, default is 'avg summer 
        temperature'
    
    Returns
    -------
    df : pandas dataframe object
        Dataframe with columns: 
            'years' - year of the observation, 
            vals_name - measurand, 
            'age' - age of the tree at the observation, 
            'file' - name of the series
            clim_name - averaged climatic value
    """
    #Reading rwl-files and inserting values to pandas dataframe
    df1 = rwl2pandas(rwl, no_data_value=no_data_value, pth_path=pth_path,
                             vals_name=vals_name)
    #Reading meteo-data
    df_meteo = read_meteo(meteo,sep=meteo_sep, clim_name=clim_name,
                          names=meteo_names, months=meteo_months)
    #Merging rwl and meteo-data by years, clim_name column could contains 
    #NaN values
    result = pd.merge(df1, df_meteo, on="years", how='left')
    result.loc[:,vals_name] = result[vals_name].astype(float).to_numpy()
    result.loc[:,'age'] = result['age'].astype(float).to_numpy()
    result.loc[:,clim_name] = result[clim_name].astype(float).to_numpy()

    return result

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

  return 1 - np.sum((targets-predictions)**2) / np.sum((targets-
                                                        np.mean(targets))**2)

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



def train_test_split(df, vals_name, clim_name, r=None, years_mask=None):
    """Splits original dataframe into train and test dataframes

      Parameters
      ----------
      df : pandas dataframe object
          dataframe in form like returned by direct_read()
      vals_name : str
          name of the column that contains proxy values
      clim_name : str
        name of the column in df that contains climatic values
      r : float, optional
          the proportion of the test sample,
          user should specify either r or years_mask
      years_mask : boolean list, optional
          boolean list for years, True for test sample,
          user should specify either r or years_mask
      
      Returns
      -------
      train: pandas dataframe object
          train part of input dataframe
      test: pandas dataframe object
          test part of input dataframe
      train_dict: dict
          dictionary with standardization parameters
    """
    if years_mask is not None:
      all_years =np.sort(np.unique(df.years))
      sel_n = [a for a, b in zip(all_years, years_mask) if b]
      test = df[df['years'].isin(sel_n)]
      train = df[~df['years'].isin(sel_n)]
    elif r is not None:
      #without respect to an order of observations
      unique_list = np.unique(df['years'])
      nr = int(r * len(unique_list))
      sel_n = np.random.choice(unique_list, nr, replace=False)
      test = df[df['years'].isin(sel_n)]
      train = df[~df['years'].isin(sel_n)]
    else:
      print('r or years_mask should be specified')

    #Standardization parameters for proxy
    train_vals_std = train[vals_name].std()
    train_vals_mean = train[vals_name].mean()
    #Standardization parameters for age
    train_age_std = train['age'].std()
    train_age_mean = train['age'].mean()
    #Standardization parameters for climatic data
    av_df = train.groupby(['years']).mean()
    train_mean_std = np.std(av_df[clim_name])
    train_mean_mean = np.mean(av_df[clim_name])
    train_dict = {'train_vals_std':train_vals_std,
                  'train_vals_mean':train_vals_mean,
                  'train_age_std':train_age_std,
                  'train_age_mean':train_age_mean,
                  'train_mean_std':train_mean_std,
                  'train_mean_mean':train_mean_mean}

    train.loc[:,'std_'+vals_name] = (train[vals_name] - 
                                     train_vals_mean) / train_vals_std
    train.loc[:,'std_age'] = (train['age'] - train_age_mean) / train_age_std
    train.loc[:,'std_'+clim_name] = (train[clim_name] - 
                                     train_mean_mean) / train_mean_std

    test.loc[:,'std_'+vals_name] = (test[vals_name] - 
                                    train_vals_mean) / train_vals_std
    test.loc[:,'std_age'] = (test['age'] - train_age_mean) / train_age_std
    test.loc[:,'std_'+clim_name] = (test[clim_name] -
                                    train_mean_mean) / train_mean_std
    
    return train, test, train_dict


def plot3d(df, Z, type_p = ['scatter', 'wireframe'], name_='',
              clim_name='avg summer temperature', vals_name='vals',
              elev=None, azim=150):
    """Creates 3d plot with axes that represent age, proxy values, and climatic 
      values

      Parameters
      ----------
      df : pandas dataframe object
          dataframe in form like returned by direct_read() or train_test_split()
      Z : 2d numpy array
          surface that was returned by sq_method()
      type_p : str list, optional
          list of graph types that will be in the figure, default is 
          ['scatter', 'wireframe'], possible options are:
                'scatter' - scatter plot with observations in df, all values in 
                clim_name column should not be NaN
                'wireframe' - wireframe that represents Z surface
                'surface' - standard visualization of Z surface
      name_ : str, optional
          name of 3d plot
      clim_name : str, optional
          name of the column in df that contains climatic values, default is 
          'avg summer temperature'
      vals_name : str, optional
          name of the column in df that contains proxy values, default is 'vals'
      elev : float, optional
          elevation parameter for 3d plot, default is None
      azim : float, optional
          azimuth parameter for 3d plot, default is 150
    """
    n = Z.shape[0] #number of values along the axis

    #axes for plot functions
    x_grid0 = np.linspace(min(df[vals_name]), max(df[vals_name])+0.1, n)
    y_grid0 = np.linspace(min(df['age']), max(df['age'])+0.1, n)
    B1, B2 = np.meshgrid(x_grid0, y_grid0, indexing='xy')

    #plots with elev specification and without
    if elev is not None:
      fig, ax = plt.subplots(subplot_kw=dict(projection='3d', azim=azim), 
                             elev=elev, 
                             gridspec_kw=dict(top=1, left=0, right=1, bottom=0),
                             figsize=(15,10))
    else:
      fig, ax = plt.subplots(subplot_kw=dict(projection='3d', azim=azim),
                        gridspec_kw=dict(top=1, left=0, right=1, bottom=0),
                        figsize=(15,10))

    ax.set_title(name_, fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    if 'wireframe' in type_p:
      ax.plot_wireframe(B1, B2, Z)
    if 'surface' in type_p:
      ax.plot_surface(B1, B2, Z, alpha=0.6)
    if 'scatter' in type_p:
      ax.scatter3D(df[vals_name], df['age'], df[clim_name], c='r', s=2)

    plt.xlabel(vals_name)
    plt.ylabel('age')
    plt.show()




def plot2d(df, Z, name_='', clim_name='avg summer temperature', 
           vals_name='vals'):
    """Creates 2d plot with axes that represent age and proxy values

      Parameters
      ----------
      df : pandas dataframe object
          dataframe in form like returned by direct_read() or train_test_split()
      Z : 2d numpy array
          surface that was returned by sq_method()
      name_ : str, optional
          name of 3d plot
      clim_name : str, optional
          name of the column in df that contains climatic values, default is 
          'avg summer temperature'
      vals_name : str, optional
          name of the column in df that contains proxy values, default is 'vals'
    """

    n = Z.shape[0] #number of values along the axis
    #axes for plot functions
    x_grid0 = np.linspace(min(df[vals_name])-1, max(df[vals_name])+1, n)
    y_grid0 = np.linspace(min(df['age']), max(df['age'])+1, n)

    # definition of 3 axs
    # axs[0][0] - 2d grid with Z values
    # axs[0][1] - colorbar (cbar1) for Z values
    # axs[1][0] - colorbar (cbar0) for climatic values in observations
    fig, axs = plt.subplots(figsize=(13, 10), nrows=2,ncols=2,
                            gridspec_kw={'height_ratios': [20,1.5],
                                         'width_ratios': [20,1]},
                            constrained_layout=True)

    axs[1][1].remove()
    im0 = axs[0][0].imshow(Z)
    vals_rescale = []
    age_rescale = []
    for v in range(len(df[vals_name])):
      # rescaling observed values for simultaneous display with Z grid
      vals_rescale0 = n * np.abs(df[vals_name].values[v]-
                                 min(x_grid0))/ np.abs(max(x_grid0)-min(x_grid0))
      age_rescale0 = n * np.abs(df['age'].values[v]-
                                min(y_grid0))/ np.abs(max(y_grid0)-min(y_grid0))
      vals_rescale.append(vals_rescale0)
      age_rescale.append(age_rescale0)

    #horizontal colorbar
    cm = plt.cm.get_cmap('YlOrBr')
    im = axs[0][0].scatter([vals_rescale], [age_rescale], 
                           c=df[clim_name], cmap=cm)
    cbar0 = fig.colorbar(im, cax=axs[1][0], orientation='horizontal')
    cbar0.set_label('observed values', fontsize=20)

    xi = list(range(len(x_grid0)))
    yi = list(range(len(y_grid0)))
    #display only every 50th elements in axes
    axs[0][0].set_xticks(ticks = xi[0::50])
    axs[0][0].set_xticklabels(np.round(x_grid0,2)[0::50])
    axs[0][0].set_yticks(ticks = yi[0::50])
    axs[0][0].set_yticklabels(np.round(y_grid0,2)[0::50])
    axs[0][0].set_ylabel('age', fontsize=20)
    axs[0][0].set_xlabel(vals_name,  fontsize=20)
    #vertical colorbar
    cbar1 = fig.colorbar(im0, cax=axs[0][1], orientation='vertical')
    cbar1.set_label('approximated values', fontsize=20)
    fig.suptitle(name_, fontsize=20)
    plt.show()


def predict_in_grid(df, grid, vals_col, age_col, vals_lim, age_lim):
    """Predicts using age and proxy data by surface

      Parameters
      ----------
      df : pandas dataframe object
          dataframe in form like returned by direct_read() or train_test_split()
      grid : 2d numpy array
          surface that was returned by sq_method()
      vals_col : str
          name of the column in df that contains proxy values
      age_col : str
          name of the column in df that contains age values
      vals_lim : float list
          list of two values that correspond to min and max values for the proxy 
          axis
      age_lim : float list
          list of two values that correspond to min and max values for the age 
          axis

      Returns
      -------
      df0: pandas dataframe object
          copy of an input df with new column 'preds'

    """
    n = grid.shape[0] #number of values along the axis
    df0 = df.copy()
    #creating axes for predictions
    x_grid0 = np.linspace(vals_lim[0], vals_lim[1], n)
    y_grid0 = np.linspace(vals_lim[1], vals_lim[0], n)
    preds = []
    for index, row in df0.iterrows():
      #finds the nearest cell in a grid with respect to observation 
      #to use it as a prediction
      x = find_nearest(x_grid0, row[vals_col])
      y = find_nearest(y_grid0, row[age_col])
      preds.append(grid[y, x])

    df0.loc[:,'preds'] = np.array(preds)
    return df0



def plot_clim_train_test(train, test, clim_var = 'avg summer temperature', 
                         ttl=''):
  """Visual comparing climatic data in test and train samples
  Expects two cases:
  1) When train and test samples are separated by date, observations in the plot 
  will be connected by a line
  2) When train and test samples are randomly separated by a ratio, observations 
  in the plot will be not connected by a line

      Parameters
      ----------
      train : pandas dataframe object
          train dataframe in form like returned by direct_read() 
          or train_test_split()
      test : 2d numpy array
          test dataframe in form like returned by direct_read() 
          or train_test_split()
      clim_var : str, optional
          name of the column in df that contains climatic values, default is 
          'avg summer temperature'
      ttl : str, optional
          name of plot
    """
  #removing observations for the same year
  tr = train.drop_duplicates(subset=['years'], ignore_index=True)
  te = test.drop_duplicates(subset=['years'], ignore_index=True)
  plt.figure(figsize=(12, 8))
  #checking for type of splitting train and test df
  if min(te.years)>max(tr.years) or min(tr.years)>max(te.years):
    #a case when train and test separated by date
    tr_m = np.repeat(np.mean(tr[clim_var]), len(tr)) #Ys for the train mean line
    te_m = np.repeat(np.mean(te[clim_var]), len(te)) #Ys for the test mean line
    plt.plot(tr['years'], tr[clim_var], 'r', label="train set")
    plt.plot(te['years'], te[clim_var], 'g', label="test set")
    plt.plot(tr['years'], tr_m, 'b', label="mean") #Xs for the train mean line
    plt.plot(te['years'], te_m, 'b') #Xs for the test mean line
    #text boxes
    txt_shift = (max(tr[clim_var])-min(tr[clim_var]))/25
    plt.text(min(tr['years']), 
            np.mean(tr[clim_var]+txt_shift), 
            np.round(np.mean(tr[clim_var]),2), 
            fontsize = 15, c='b',
            bbox = dict(facecolor = 'white', alpha = 0.8)) #train mean text
    plt.text(min(te['years']), 
            np.mean(te[clim_var]+txt_shift), 
            np.round(np.mean(te[clim_var]),2), 
            fontsize = 15, c='b',
            bbox = dict(facecolor = 'white', alpha = 0.8)) #test mean text
  else:
    #a case when train and test are randomly separated by ratio
    plt.plot(tr['years'], tr[clim_var], 'ro', label="train set")
    plt.plot(te['years'], te[clim_var], 'go', label="test set")
    tr_mean = np.round(np.mean(tr[clim_var]),2)
    te_mean = np.round(np.mean(te[clim_var]),2)
    #Mean values will be printed in a legend
    plt.plot([], [], ' ', label="train_mean = " + str(tr_mean))
    plt.plot([], [], ' ', label="test_mean = " + str(te_mean))

  plt.xlabel('years',fontsize=15)
  plt.ylabel(clim_var, fontsize=15)
  plt.legend(loc="lower right", prop={'size': 15})
  plt.show()



def sq_method(train, train_dict, n, n_sq, 
              vals_col, age_col, vals_lim, age_lim,
              smooth=5, thr_cou=0,
              clim_var = 'avg summer temperature',
              square_plot = True,
              use_std=True):
      """Creates an approximation surface for climatic values prediction by
      following steps:
      1) Averaging train observations that get into a cell of a regular 
      square grid
      2) Constructing a thin-plate-spline surface with 
      scipy.interpolate.RBFInterpolator()

          Parameters
          ----------
          train : pandas dataframe object
              dataframe in form like returned by direct_read() or 
              train_test_split() with train observations
          train_dict : dict
              dictionary with standardization parameters
          n : int
              length of elements in every axis of the resulting surface
          n_sq : int
              number of averaging squares on every axis
          vals_col : str
              name of the column in df that contains proxy values
          age_col : str
              name of the column in train that contains age values
          vals_lim : float list
              list of two values that correspond to min and max values for 
              the proxy axis
          age_lim : float list
              list of two values that correspond to min and max values for 
              the age axis
          smooth : float, optional
              smoothing parameter for scipy.interpolate.RBFInterpolator(),
              default is 5
          thr_cou : int, optional
              threshold value for number of observations squares in first step,
              default is 0
          clim_var : str, optional
              name of the column in df that contains climatic values, default is
              'avg summer temperature'
          square_plot : boolean, optional
              if True, plots heatmap of mean values in cells before 
              constructing a surface, default is True
          use_std: boolean, optional
              to use standardized axes for surface construction or use original 
              axes instead, default is True
          
          Returns
          -------
          Z: 2d numpy array
              approximation surface for climatic values

      """
      #Creating empty 2d arrays for results
      Z_std = np.empty([n_sq, n_sq])
      Z_std[:] = np.nan
      cou_Z = np.empty([n_sq, n_sq])
      cou_Z[:] = np.nan
      mdf = train.copy()

      #axes for approximation surface
      x_grid0 = np.linspace(vals_lim[0], vals_lim[1], n)
      y_grid0 = np.linspace(age_lim[0], age_lim[1], n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')
      #axes for averaging train observations
      x_grid = np.linspace(vals_lim[0], vals_lim[1], n_sq)
      y_grid = np.linspace(age_lim[0], age_lim[1], n_sq)
      B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

      Z_list = []
      xcou = 0
      #step sizes for averaging
      st_x = x_grid[1]-x_grid[0]
      st_y = y_grid[1]-y_grid[0]
      for xi in x_grid:
        ycou = 0
        #selecting observations in row
        mdf_x = mdf[(mdf[vals_col] >= xi-0.5*st_x) & 
                    (mdf[vals_col] < xi+0.5*st_x)]
        for yi in y_grid:
            #selecting observations in cell
            mdf_xy = mdf_x[(mdf_x[age_col] >= yi-0.5*st_y) & 
                           (mdf_x[age_col] < yi+0.5*st_y)]
            #checking the number of observations in cell
            if len(mdf_xy) > thr_cou: 
                Z_std[xcou, ycou] = np.mean(mdf_xy['std_' + clim_var])
                #counting observations for square_plot
                cou_Z[xcou, ycou] = len(mdf_xy) 

            ycou = ycou + 1
        xcou = xcou + 1

      #organizes the data for sp.interpolate.RBFInterpolator()
      zr = Z_std.ravel()[~np.isnan(Z_std.ravel())]
      xr = B1.ravel()[~np.isnan(Z_std.ravel())]
      yr = B2.ravel()[~np.isnan(Z_std.ravel())]
      xx = np.array([xr,yr]).T
      xx0 = np.array([B1_0.ravel(), B2_0.ravel()]).T
      #thin-plate-spline approximation
      yflat = sp.interpolate.RBFInterpolator(xx, zr, smoothing=smooth)(xx0)
      ydata_s = yflat.reshape(n, n)
      #2d plot with mean values in cells
      if square_plot:
        name_='n_squares = '+ str(n_sq) # title
        fig, axs = plt.subplots(figsize=(13, 10))
        if use_std:
          #standardization
          y_grid = (y_grid*train_dict['train_age_std']+
                    train_dict['train_age_mean'])
          x_grid = (x_grid*train_dict['train_vals_std']+
                    train_dict['train_vals_mean'])
        #heatmap with number of observation in every cell
        axs = sns.heatmap(Z_std*train_dict['train_mean_std']+
                          train_dict['train_mean_mean'], 
                          cmap="jet", annot=np.int16(cou_Z), 
                          annot_kws={'fontsize': 9},fmt = '',
                          xticklabels = np.round(y_grid,1),
                          yticklabels = np.round(x_grid,1))
        axs.set_title(name_, fontsize=20)

      #restandardization of climatic values
      return (ydata_s.T*train_dict['train_mean_std']+
              train_dict['train_mean_mean'])


def plot_preds(pred_df, y_lables_step=5):
  """Creates 2d plot with predictions where columns correspond to years and 
      rows correspond to tree series

      Parameters
      ----------
      pred_df : pandas dataframe object
          dataframe in form like returned by direct_read() or train_test_split()
          with 'preds' column
      y_lables_step : int, optional
          step for displaying y-axis labels, defauilt is 5
  """
  df = pred_df.copy()
  min_year = np.min(df['years'].to_numpy())
  max_year = np.max(df['years'].to_numpy())
  #group by name of a series
  g_df = df.groupby('file').agg({'years':'max', 'preds':'count'}).reset_index()
  #sorting table in a way where at the top of the plot is the most 
  #recent predictions
  g_df = g_df.sort_values(by=['years','preds'], ascending = [False, True], 
                          ignore_index=True)

  un_f = g_df['file'].to_numpy() #array with name of series
  #creation empty 2d array for plot
  res2d = np.empty((len(un_f), max_year-min_year+1))
  res2d[:] = np.nan
  cou = 0
  #iteration over series
  for u in un_f:
    t_df = df[df['file']==u]
    for index, row in t_df.iterrows():
      #year of prediction to column index of 2d array
      year_ind0 = row['years']-min_year
      res2d[cou, year_ind0] = row['preds']

    cou += 1


  fig, axs = plt.subplots(figsize=(10, 15))
  im0 = plt.imshow(res2d)
  #colorbar for res2d
  cbar0 = plt.colorbar(im0, orientation='vertical',fraction=0.046, pad=0.04)
  cbar0.set_label('predicted values', fontsize=20)
  axs.set_ylabel('tree', fontsize=20)
  axs.set_xlabel('year',  fontsize=20)
  yi = list(range(len(un_f)))
  axs.set_yticks(ticks = yi[0::y_lables_step])
  axs.set_yticklabels(un_f[0::y_lables_step])
  #year labels with 10-years steps
  axs.set_xticks(ticks = list(range(-min_year+max_year))[0::10])
  axs.set_xticklabels(list(range(min_year,max_year))[0::10])

  plt.show()




def direct(train, train_dict, test, 
           sm, n_sq, n,
           use_std=True,
           vals_name='MXD', 
           clim_var = 'avg summer temperature',
           square_plot = True,
           uncertainty_data_rep = None,
           uncertainty_instrumental = None):
  """Creates an approximation surface for climatic values prediction 
  and performs prediction

      Parameters
      ----------
          train : pandas dataframe object
              dataframe in form like returned by direct_read() or 
              train_test_split() with train observations
          train_dict : dict
              dictionary with standardization parameters
          test : pandas dataframe object
              dataframe in form like returned by direct_read() or 
              train_test_split() with test observations
          sm : float
              smoothing parameter for thin-plate-spline 
              (scipy.interpolate.RBFInterpolator())
          n_sq : int
              number of averaging squares on every axis (used in sq_method())
          n : int
              length of elements in every axis of the resulting surface
          use_std: boolean, optional
              to use standardized axes for surface construction or use original 
              axes instead, default is True
          vals_name : str, optional
              name of the column in df that contains proxy values,
              default is 'MXD'
          clim_var : str, optional
              name of the column in df that contains climatic values, default is
              'avg summer temperature'
          square_plot : boolean, optional
              if True, plots heatmap of mean values in suppliment cells before 
              constructing a surface, default is True
          uncertainty_data_rep: dict, optional
              computing the uncertainty of data replication, default is None
              dict consists of the following parameters:
                  "part_s" - part of samples that will be picked up at 
                  bootstrapping procedure,
                  "n_iter" - number of iterations at bootstrapping,
                  "alpha" - significance level
              if uncertainty_data_rep is None, envelops does not compute
          uncertainty_instrumental: dict, optional
              computing the instrumental uncertainty, default is None
              dict consists of the following parameters:
                  "part_s" - part of samples that will be picked up at 
                  bootstrapping procedure,
                  "n_iter" - number of bootstrapping iterations,
                  "alpha" - significance level
              if uncertainty_instrumental is None, envelops does not compute
          
      Returns
      -------
          pred_df: pandas dataframe object
              copy of a test df with new column 'preds' and
              columns 'lower_1', 'upper_1', if uncertainty_data_rep is 
              specified, and columns 'lower_2', 'upper_2', 
              if uncertainty_instrumental is specified
          Z_sq_method : 2d numpy array
              approximation surface for climatic values
  """
  if use_std:
        vals_col = 'std_' + vals_name
        age_col = 'std_age'
  else:
        vals_col = vals_name
        age_col = 'age'

  #bounds of approximation surface selects as the most extreme values in test 
  #and train
  vals_lim0 = min([min(train[vals_col]), min(test[vals_col])])-0.01
  vals_lim1 = max([max(train[vals_col]), max(test[vals_col])])+0.01
  age_lim0 = min([min(train[age_col]), min(test[age_col])])-0.01
  age_lim1 = max([max(train[age_col]), max(test[age_col])])+0.01
  vals_lim = [vals_lim0,vals_lim1]
  age_lim = [age_lim0,age_lim1]

  #constructing an approximation surface                        
  Z_sq_method = sq_method(train, train_dict, n, n_sq, smooth=sm,
                          vals_col = vals_col, 
                          age_col = age_col,
                          vals_lim = vals_lim, 
                          age_lim = age_lim,
                          clim_var = clim_var,
                          square_plot = square_plot,
                          use_std=use_std)
  #predicting on another df
  pred_df = predict_in_grid(test, Z_sq_method, vals_col, age_col,
                            vals_lim = vals_lim, age_lim = age_lim)
  #Variance adjustment
  #averaging train df by years
  av_df = train.groupby(['years']).mean()
  train_std = np.std(av_df[clim_var]) #train climatic data std
  #averaging predictions df by years
  av_df_pr = pred_df.groupby(['years']).mean()
  pred_mean = np.mean(av_df_pr['preds']) #mean of predictions
  pred_std = np.std(av_df_pr['preds']) #std of predictions
  std_rat = train_std/pred_std
  pred_df.loc[:,'preds'] = (pred_df['preds']-pred_mean)*std_rat+pred_mean

  if uncertainty_data_rep is not None:
    print('computing the uncertainty of data replication')
    lower_1 = []
    upper_1 = []
    years = []
    #itearting by years in pred_df
    for y in tqdm(np.unique(pred_df['years'])):
      tt = pred_df[pred_df['years']==y] #selecting particular year
      part_s = uncertainty_data_rep['part_s']
      n_size = int(part_s*len(tt))
      n_iter = uncertainty_data_rep['n_iter']
      stats = [] #list for means
      for i in range(n_iter):
        #random selection n_size elements
        res = resample(tt['preds'], n_samples=n_size) 
        stats.append(np.mean(res))

      # confidence intervals
      alpha = uncertainty_data_rep['alpha']
      p = alpha/2.0
      lower = np.quantile(stats, p)
      p = 1 - alpha+(alpha/2.0)
      upper = np.quantile(stats, p)
      #bounds of envelope
      lower_1.append(lower)
      upper_1.append(upper)
      years.append(y)

    u_data_rep = pd.DataFrame([years, lower_1, upper_1], 
                              index=['years', 'lower_1', 'upper_1']).T
    u_data_rep.loc[:,'lower_1'] = u_data_rep['lower_1']
    u_data_rep.loc[:,'upper_1'] = u_data_rep['upper_1']
    #merging dataframes with predictions and envelope bounds
    pred_df = pd.merge(pred_df, u_data_rep, on="years", how='left')

  if uncertainty_instrumental is not None:
    print('computing the uncertainty of instrumental period')
    lower_2 = []
    upper_2 = []
    pred_list = []
    for i in tqdm(range(uncertainty_instrumental['n_iter'])):
      #performing surface construction and prediction n_iter times
      #on randomly selected years in train df
      yy = np.unique(train['years'])
      part_s = uncertainty_instrumental['part_s']
      n_size = int(part_s*len(yy))
      
      res = resample(yy, n_samples=n_size)
      tt = train[train['years'].isin(res)] #selecting years
      #constructing an approximation surface  
      Z_sq_method_t = sq_method(tt, train_dict, n, n_sq, smooth=sm,
                          vals_col = vals_col, 
                          age_col = age_col,
                          vals_lim = vals_lim, 
                          age_lim = age_lim,
                          clim_var = clim_var,
                          square_plot = False,
                          use_std=use_std)
      #predicting on another df
      pred_df_t = predict_in_grid(test, Z_sq_method_t, n, vals_col, age_col,
                                  vals_lim = vals_lim, age_lim = age_lim)
      #Variance adjustment
      av_df_pr = pred_df_t.groupby(['years']).mean()
      av_df = tt.groupby(['years']).mean()
      train_std = np.std(av_df[clim_var])
      pred_mean = np.mean(av_df_pr['preds'])
      pred_std = np.std(av_df_pr['preds'])
      std_rat = train_std/pred_std
      pred_df_t.loc[:,'preds'] = (pred_df_t['preds']-
                                  pred_mean)*std_rat+pred_mean
      av_df_pr = pred_df_t.groupby(['years']).mean()
      pred_list.append(av_df_pr['preds'].values)

    years = av_df_pr.index.values
    stats = np.array(pred_list)
    # confidence intervals
    alpha = uncertainty_instrumental['alpha']
    p = alpha/2.0
    lower = np.quantile(stats, p, axis=0)
    p = 1 - alpha+(alpha/2.0)
    upper = np.quantile(stats, p, axis=0)

    u_uncertainty_instrumental = pd.DataFrame([years, lower, upper], 
                                              index=['years', 'lower_2', 
                                                     'upper_2']).T
    #merging dataframes with predictions and envelope bounds
    pred_df = pd.merge(pred_df, u_uncertainty_instrumental, 
                       on="years", how='left')

  return pred_df, Z_sq_method


def parameter_plot(train, train_dict, test, n, clim_var, vals_name, n_sq=30, 
                   sm=5, use_std = True):
  """Creates 2d heatmap for different values of smoothing and number of 
  supplementary squares; or graph if one of them specified as a single value.
  Metrics that used: 'Correlation', 'CE', 'RMSE'

      Parameters
      ----------
          train : pandas dataframe object
              dataframe in form like returned by direct_read() or 
              train_test_split() with train observations
          train_dict : dict
              dictionary with standardization parameters
          test : pandas dataframe object
              dataframe in form like returned by direct_read() or 
              train_test_split() with test observations
          n : int
              length of elements in every axis of the resulting surface
          clim_var : str
              name of the column in df that contains climatic values
          vals_name : str
              name of the column in df that contains proxy values
          n_sq : int or int list, optional
              number of averaging squares on every axis (used in sq_method()),
              default is 30
          sm : float or float list, optional
              smoothing parameter for thin-plate-spline 
              (scipy.interpolate.RBFInterpolator()), default is 5
          use_std: boolean, optional
              to use standardized axes for surface construction or use original 
              axes instead, default is True
          
      Returns
      -------
          metrics : list of numpy arrays 
              a list that contains computed metrics: 'Correlation','CE', 'RMSE'
  """
  if isinstance(sm, list) and isinstance(n_sq, list):
    #if n_sq and sm are lists, computes 2d array for each metric
    #empty 2d arrays for metrics
    metrics_corr = np.empty([len(n_sq), len(sm)])
    metrics_corr[:] = np.nan
    metrics_CE = np.empty([len(n_sq), len(sm)])
    metrics_CE[:] = np.nan
    metrics_RMSE = np.empty([len(n_sq), len(sm)])
    metrics_RMSE[:] = np.nan
    #loop on n_sq (displays as progress bar)
    for i in tqdm(range(len(n_sq))):
        #loop om sm
        for j in range(len(sm)):                             
          pred_df, Z_sq_method = direct(train, train_dict, test, 
                                        sm[j], n_sq[i], n,
                                        vals_name=vals_name, 
                                        clim_var = clim_var,
                                        square_plot = False, use_std=use_std
                                        )
          #averaging predictions by years
          av_df = pred_df.groupby(['years']).mean()
          #corr
          corr_Spearman = sp.stats.spearmanr(av_df[clim_var], av_df['preds'])[0]
          metrics_corr[i,j] = corr_Spearman
          #CE
          CE_ = CE(av_df[clim_var],av_df['preds'])
          metrics_CE[i,j] = CE_
          #RMSE
          RMSE = ((av_df[clim_var] - av_df['preds']) ** 2).mean() ** .5
          metrics_RMSE[i,j] = RMSE

    metrics = [metrics_corr, metrics_CE, metrics_RMSE]
    names = ['Correlation','CE', 'RMSE']
    #heatmap for each metrics
    for met in range(len(metrics)):
      fig, axs = plt.subplots(figsize=(20, 10))
      axs = sns.heatmap(metrics[met], 
                            cmap="jet", annot=np.round(metrics[met],2), 
                            annot_kws={'fontsize': 9},fmt = '',
                            xticklabels = np.round(sm,1),
                            yticklabels = np.round(n_sq,1))
      plt.xlabel('smooth')
      plt.ylabel('n squares')
      axs.set_title(names[met], fontsize = 15)
      plt.show()
    
  else:
    #line graph with one varible
    metrics_corr = []
    metrics_CE = []
    metrics_RMSE = []
    if isinstance(sm, list):
      iter_l = sm
    else:
      iter_l = n_sq

    for i in tqdm(iter_l):
        if isinstance(sm, list):
          #sm is a list                             
          pred_df, Z_sq_method = direct(train, train_dict, test,
                                        i, n_sq, 
                                        n,
                                        vals_name=vals_name, 
                                        clim_var = clim_var,
                                        square_plot = False, use_std=use_std)
          x_axs_name = 'smooth'
        else:
          #n_sq is a list
          pred_df, Z_sq_method = direct(train, train_dict, test,
                                        sm, i, 
                                        n,
                                        vals_name=vals_name, 
                                        clim_var = clim_var,
                                        square_plot = False, use_std=use_std)
          x_axs_name = 'n squares'

        #averaging predictions by years  
        av_df = pred_df.groupby(['years']).mean()
        #corr
        corr_Spearman = np.round(sp.stats.spearmanr(av_df[clim_var], 
                                                    av_df['preds'])[0],3)
        metrics_corr.append(corr_Spearman)
        #CE
        CE_ = CE(av_df[clim_var],av_df['preds'])
        metrics_CE.append(CE_)
        #RMSE
        RMSE = ((av_df[clim_var] - av_df['preds']) ** 2).mean() ** .5
        metrics_RMSE.append(RMSE)

    metrics = [metrics_corr, metrics_CE, metrics_RMSE]
    names = ['Correlation','CE', 'RMSE']
    #line graph for each metrics
    for met in range(len(metrics)):
      plt.figure(figsize=(12, 8))
      plt.plot(iter_l, metrics[met], label = names[met])
      plt.xlabel(x_axs_name,fontsize=15)
      plt.ylabel(names[met], fontsize=15)
      plt.legend(fontsize=15)
      if names[met] != 'RMSE':
        index_m = np.argmax(np.array(metrics[met]))
      else:
        index_m = np.argmin(np.array(metrics[met]))

      plt.annotate(str(iter_l[index_m]),(iter_l[index_m],metrics[met][index_m]),color='red')
      plt.show()
  
  return metrics
