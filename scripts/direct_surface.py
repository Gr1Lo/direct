import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import seaborn as sns
import warnings
from sklearn.utils import resample
from tqdm import tqdm

from read_data import rwl2pandas, read_meteo, direct_read, standardize_train, standardize_test, train_test_split
from utils import find_nearest, CE, RE, direct_statistics

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams['figure.dpi'] = 100

def sq_method(train, train_dict, n, n_sq,
              proxy_col, age_col, proxy_lim, age_lim,
              smooth=5, kernel='thin_plate_spline', thr_cou=0,
              clim_var = 'avg summer temperature',
              square_plot = True,
              use_std=True,
              use_squares=False):
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
          proxy_col : str
              name of the column in df that contains proxy values
          age_col : str
              name of the column in train that contains age values
          proxy_lim : float list
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
          use_squares : boolean, optional
              if True, constructs surface by averaged in squares data,
              default is False

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
      x_grid0 = np.linspace(proxy_lim[0], proxy_lim[1], n)
      y_grid0 = np.linspace(age_lim[0], age_lim[1], n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')
      #axes for averaging train observations
      x_grid = np.linspace(proxy_lim[0], proxy_lim[1], n_sq)
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
        mdf_x = mdf[(mdf[proxy_col] >= xi-0.5*st_x) &
                    (mdf[proxy_col] < xi+0.5*st_x)]
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

      #organizing the data for sp.interpolate.RBFInterpolator()
      xx0 = np.array([B1_0.ravel(), B2_0.ravel()]).T

      #rbf approximation
      if use_squares:
        zr = Z_std.ravel()[~np.isnan(Z_std.ravel())]
        xr = B1.ravel()[~np.isnan(Z_std.ravel())]
        yr = B2.ravel()[~np.isnan(Z_std.ravel())]
        xx = np.array([xr,yr]).T
        yflat = sp.interpolate.RBFInterpolator(xx, zr, smoothing=smooth, kernel=kernel)(xx0)
      else:
        #xx0 = np.array([B1_0.ravel(), B1_0.ravel()]).T
        xx = np.array([mdf[age_col].values,mdf[proxy_col].values]).T
        #xx = np.array([mdf['std_age'].values,mdf['std_proxy'].values]).T
        zr = mdf['std_' + clim_var].values
        yflat = sp.interpolate.RBFInterpolator(xx, zr, smoothing=smooth, kernel=kernel)(xx0)#, degree=-1)(xx0)

      ydata_s = yflat.reshape(n, n)
      #2d plot with mean values in cells
      if square_plot:
        name_='n_squares = '+ str(n_sq) # title
        fig, axs = plt.subplots(figsize=(13, 10))
        if use_std:
          #standardization
          y_grid = (y_grid*train_dict['train_age_std']+
                    train_dict['train_age_mean'])
          x_grid = (x_grid*train_dict['train_proxy_std']+
                    train_dict['train_proxy_mean'])
        #heatmap with number of observation inside each cell
        axs = sns.heatmap(Z_std*train_dict['train_mean_std']+
                          train_dict['train_mean_mean'],
                          cmap="jet", annot=np.int16(cou_Z),
                          annot_kws={'fontsize': 9},fmt = '',
                          xticklabels = np.round(y_grid,1),
                          yticklabels = np.round(x_grid,1))

        ageN = age_col.replace('std_','')
        proxyN = proxy_col.replace('std_','')
        axs.set(xlabel=ageN, ylabel=proxyN)
        axs.set_title(name_, fontsize=20)

      #restandardization of climatic values
      return (ydata_s.T*train_dict['train_mean_std']+ train_dict['train_mean_mean'])







def make_surface(train, train_dict,
                 sm, n_sq, n,
                 proxy_lim=[None,None], age_lim=[None,None],
                 use_std=True, kernel='thin_plate_spline',
                 proxy_name='MXD',age_var = 'age',
                 clim_var = 'avg summer temperature',
                 square_plot = True,
                 use_squares=False):
  """Creates an approximation surface for climatic values prediction

      Parameters
      ----------
          train : pandas dataframe object
              dataframe in form like returned by direct_read() or
              train_test_split() with train observations
          train_dict : dict
              dictionary with standardization parameters
          sm : float
              smoothing parameter for thin-plate-spline
              (scipy.interpolate.RBFInterpolator())
          n_sq : int
              number of averaging squares on every axis (used in sq_method())
          n : int
              length of elements in every axis of the resulting surface
          proxy_lim : list, optional
              boundaries of surface in proxy axis, default is [None,None]
              if [None,None], computes by min and max values of train df
          age_lim : list, optional
              boundaries of surface in age axis, default is [None,None]
              if [None,None], computes by min and max values of train df
          use_std: boolean, optional
              to use standardized axes for surface construction or use original
              axes instead, default is True
          proxy_name : str, optional
              name of the column in df that contains proxy values,
              default is 'MXD'
          clim_var : str, optional
              name of the column in df that contains climatic values, default is
              'avg summer temperature'
          square_plot : boolean, optional
              if True, plots heatmap of mean values in suppliment cells before
              constructing a surface, default is True
          use_squares : boolean, optional
              if True, constructs surface by averaged in squares data,
              default is False

      Returns
      -------
          surface : 2d numpy array
              approximation surface for climatic values
  """

  if use_std:
        proxy_col = 'std_' + proxy_name
        age_col = 'std_' + age_var
  else:
        proxy_col = proxy_name
        age_col = age_var

  proxy_lim0 = proxy_lim.copy()
  age_lim0 = age_lim.copy()

  if use_std:
        proxy_lim0[0] = (proxy_lim0[0]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
        proxy_lim0[1] = (proxy_lim0[1]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
        age_lim0[0] = (age_lim0[0]-train_dict['train_age_mean']) / train_dict['train_age_std']
        age_lim0[1] = (age_lim0[1]-train_dict['train_age_mean']) / train_dict['train_age_std']

  #constructing an approximation surface
  surface = sq_method(train, train_dict, n, n_sq, smooth=sm,
                          proxy_col = proxy_col,
                          age_col = age_col,
                          proxy_lim = proxy_lim0,
                          age_lim = age_lim0,
                          kernel=kernel,
                          clim_var = clim_var,
                          square_plot = square_plot,
                          use_std=use_std,
                          use_squares=use_squares)

  return surface

def predict_on_surface(surface, train_dict, train, test, 
                       proxy_lim,
                       age_lim,
                       use_std=True,
                       proxy_name='MXD', age_var = 'age',
                       clim_var = 'avg summer temperature',
                       uncertainty_data_rep = None,
                       uncertainty_instrumental = None,
                       Z_shift=True):
  """Performs a climatic values prediction

      Parameters
      ----------
          surface : 2d numpy array
              approximation surface for climatic values
          train_dict : dict
              dictionary with standardization parameters
          test : pandas dataframe object
              dataframe in form like returned by direct_read() or
              train_test_split() with test observations
          proxy_lim : list, optional
              boundaries of surface in proxy axis, default is [None,None]
              if [None,None], computes by min and max values of train df
          age_lim : list, optional
              boundaries of surface in age axis, default is [None,None]
              if [None,None], computes by min and max values of train df
          use_std: boolean, optional
              to use standardized axes for surface construction or use original
              axes instead, default is True
          proxy_name : str, optional
              name of the column in df that contains proxy values,
              default is 'MXD'
          clim_var : str, optional
              name of the column in df that contains climatic values, default is
              'avg summer temperature'
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
                  "alpha" - significance level,
                  "kernel" - smoothing kernel for constucting surface,
                  "use_squares" - using squares or not, boolean
              if uncertainty_instrumental is None, envelops does not compute
          Z_shift: boolean, optional, default is True

      Returns
      -------
          pred_df: pandas dataframe object
              copy of a test df with new column 'preds' and
              columns 'lower_1', 'upper_1', if uncertainty_data_rep is
              specified, and columns 'lower_2', 'upper_2',
              if uncertainty_instrumental is specified
  """

  #defines which columns to use
  if use_std:
        proxy_col = 'std_' + proxy_name
        age_col = 'std_' + age_var
  else:
        proxy_col = proxy_name
        age_col = age_var


  proxy_lim0 = proxy_lim.copy()
  age_lim0 = age_lim.copy()
  if use_std:
        proxy_lim0[0] = (proxy_lim0[0]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
        proxy_lim0[1] = (proxy_lim0[1]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
        age_lim0[0] = (age_lim0[0]-train_dict['train_age_mean']) / train_dict['train_age_std']
        age_lim0[1] = (age_lim0[1]-train_dict['train_age_mean']) / train_dict['train_age_std']

  n_gr = surface.shape[0] #number of values along the axis
  pred_df = test.copy()
  #creating axes for predictions
  x_grid0 = np.linspace(proxy_lim0[0], proxy_lim0[1], n_gr)
  y_grid0 = np.linspace(age_lim0[0], age_lim0[1], n_gr)
  preds = []

  preds_train = []
  for index, row in pred_df.iterrows():
      #finds the nearest cell in a grid with respect to observation
      #to use it as a prediction
      x = find_nearest(x_grid0, row[proxy_col])
      y = find_nearest(y_grid0, row[age_col])
      preds.append(surface[y, x])

  for index, row in train.iterrows():
      y_train = find_nearest(x_grid0, row[proxy_col])
      x_train = find_nearest(y_grid0, row[age_col])
      preds_train.append(surface[y_train, x_train])

  train.loc[:,'preds'] = np.array(preds_train)

  #==========Z_shift

  if Z_shift:
    x_mean = find_nearest(x_grid0, np.mean(pred_df[proxy_col]))
    y_mean = find_nearest(y_grid0, np.mean(pred_df[age_col]))
    mean_Z_test = surface[y_mean, x_mean]

    x_mean_train = find_nearest(x_grid0, np.mean(train[proxy_col]))
    y_mean_train = find_nearest(y_grid0, np.mean(train[age_col]))
    mean_Z_train = surface[y_mean_train, x_mean_train]

    Z_shift = mean_Z_test-mean_Z_train

  else:
    Z_shift=0

  #================================

  pred_df.loc[:,'preds'] = np.array(preds)

  #Variance adjustment
  #averaging train df by years
  av_df = train.groupby(['years']).mean(numeric_only=True)
  train_std = np.std(av_df[clim_var])#train climatic data std
  train_mean = np.mean(av_df[clim_var])
  #pred_std = np.std(av_df['preds'])

  #averaging predictions df by years
  av_df_pr = pred_df.groupby(['years']).mean(numeric_only=True)
  pred_mean = np.mean(pred_df['preds'])
  av_pred_mean = np.mean(av_df_pr['preds'])
  pred_std = np.std(av_df_pr['preds']) #std of predictions
  std_rat = (train_std/pred_std)

  av_df_pr.loc[:,'preds_unscaled'] = av_df_pr['preds']+ Z_shift
  av_df_pr.loc[:,'preds'] = (av_df_pr['preds']-av_pred_mean)*std_rat+train_mean#+av_pred_mean + Z_shift

  if uncertainty_data_rep is not None:
    print('computing the uncertainty of data replication')
    lower_1 = []
    upper_1 = []
    median_L=[]
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

      # confidence
      alpha = uncertainty_data_rep['alpha']
      p = alpha/2.0
      lower = np.quantile(stats, p)
      p = 1 - alpha+(alpha/2.0)
      upper = np.quantile(stats, p)
      median = np.quantile(stats, 0.5)
      #bounds of envelope
      lower_1.append((lower-av_pred_mean)*std_rat+av_pred_mean + Z_shift)
      upper_1.append((upper-av_pred_mean)*std_rat+av_pred_mean + Z_shift)
      median_L.append((median-av_pred_mean)*std_rat+av_pred_mean + Z_shift)
      years.append(y)

    u_data_rep = pd.DataFrame([years, lower_1, upper_1, median_L],
                              index=['years', 'lower_1', 'upper_1', 'median']).T
    u_data_rep.loc[:,'lower_1'] = u_data_rep['lower_1']
    u_data_rep.loc[:,'upper_1'] = u_data_rep['upper_1']
    u_data_rep.loc[:,'median'] = u_data_rep['median']
    #merging dataframes with predictions and envelope bounds
    av_df_pr = pd.merge(av_df_pr, u_data_rep, on="years", how='left')


  if uncertainty_instrumental is not None:
    print('computing the uncertainty of instrumental period')
    n = uncertainty_instrumental['n']
    n_sq = uncertainty_instrumental['n_sq']
    sm = uncertainty_instrumental['sm']
    proxy_lim0 = proxy_lim.copy()
    age_lim0 = age_lim.copy()
    if use_std:
            proxy_lim0[0] = (proxy_lim0[0]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
            proxy_lim0[1] = (proxy_lim0[1]-train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
            age_lim0[0] = (age_lim0[0]-train_dict['train_age_mean']) / train_dict['train_age_std']
            age_lim0[1] = (age_lim0[1]-train_dict['train_age_mean']) / train_dict['train_age_std']

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
      surface_t = sq_method(tt, train_dict, n, n_sq, smooth=sm,
                          proxy_col = proxy_col,
                          age_col = age_col,
                          proxy_lim = proxy_lim0,
                          age_lim = age_lim0,
                          kernel=uncertainty_instrumental['kernel'],
                          clim_var = clim_var,
                          square_plot = False,
                          use_std=use_std,
                          use_squares=uncertainty_instrumental['use_squares'])

      #predicting on another df

      n_gr = surface_t.shape[0] #number of values along the axis
      pred_df_t = test.copy()
      #creating axes for predictions
      x_grid0 = np.linspace(proxy_lim0[0], proxy_lim0[1], n_gr)
      y_grid0 = np.linspace(age_lim0[0], age_lim0[1], n_gr)
      preds = []
      for index, row in pred_df.iterrows():
          #finds the nearest cell in a grid with respect to observation
          #to use it as a prediction
          x = find_nearest(x_grid0, row[proxy_col])
          y = find_nearest(y_grid0, row[age_col])
          preds.append(surface_t[y, x])

      pred_df_t.loc[:,'preds'] = np.array(preds)


      #Variance adjustment
      av_df = tt.groupby(['years']).mean(numeric_only=True)
      train_std = np.std(av_df[clim_var])
      #pred_std = np.std(av_df['preds'])

      av_df_pr = pred_df_t.groupby(['years']).mean(numeric_only=True)
      pred_mean = np.mean(av_df_pr['preds'])
      av_pred_mean = np.mean(av_df_pr['preds'])
      pred_std = np.std(av_df_pr['preds']) #std of predictions
      std_rat = train_std/pred_std

      av_df_pr.loc[:,'preds'] = (av_df_pr['preds']-av_pred_mean)*std_rat+av_pred_mean + Z_shift
      pred_list.append(av_df_pr['preds'].values)

    years = av_df_pr.index.values
    stats = np.array(pred_list)
    # confidence interproxy
    alpha = uncertainty_instrumental['alpha']
    p = alpha/2.0
    lower = np.quantile(stats, p, axis=0)
    p = 1 - alpha+(alpha/2.0)
    upper = np.quantile(stats, p, axis=0)

    median = np.quantile(stats, 0.5, axis=0)

    u_uncertainty_instrumental = pd.DataFrame([years, lower, upper, median],
                                              index=['years', 'lower_2',
                                                     'upper_2','median']).T
    #merging dataframes with predictions and envelope bounds
    av_df_pr = pd.merge(av_df_pr, u_uncertainty_instrumental,
                       on="years", how='left')

  return av_df_pr




def leave_k_out_plot(df, clim_name, proxy_name, age_name='age',
                     k=20, n=150,
                     n_sq=None, sm=None,
                     kernel='linear', use_std = True, use_squares=True, Z_shift=True):
    """Creates 2d heatmap for different values of smoothing and number of
    supplementary squares. Can be use for finding best n_sq and sm values.
    Metrics that used: 'Correlation', 'CE'

        Parameters
        ----------
            df : pandas dataframe object
                dataframe in form like returned by direct_read() or
                train_test_split() with train observations
            clim_name : str
                name of the column in df that contains climatic values
            proxy_name : str
                name of the column in df that contains proxy values
            k : int
                number of years that removed from training dataset at every step,
                default is 20
            n : int
                length of elements in every axis of the resulting surface,
                default is 150
            n_sq : int or int list, optional
                number of averaging squares on every axis (used in sq_method()),
                default is 30
            sm : float or float list, optional
                smoothing parameter for thin-plate-spline
                (scipy.interpolate.RBFInterpolator()), default is 5
            kernel : str
                kernel for smoothing, default is 'linear'
            use_std: boolean, optional
                to use standardized axes for surface construction or use original
                axes instead, default is True
            use_squares : boolean, optional
                if True, constructs surface by averaged in squares data,
                default is False

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
      for j in tqdm(range(len(sm))):
          print('smooth: ' + str(sm[j]))
          #loop on sm
          for i in tqdm(range(len(n_sq))):
              print('n squares: ' + str(n_sq[i]))
              dft_ys = df.loc[~df[clim_name].isna()]
              ys0=np.arange(min(dft_ys.years),max(dft_ys.years)+1)
              ys=np.arange(min(df.years),max(df.years)+1)
              origs = []
              preds = []
              #loop on years
              for year in ys0:
                add_rb = 0
                target_year_mask = (ys==year)
                mask = target_year_mask.copy()
                lb = np.where(mask==True)[0][0]-int(k/2)
                if lb<0:
                      add_rb = -lb
                      lb=0#?

                rb = np.where(mask==True)[0][0]+int(k/2) +1 + add_rb
                if rb>len(mask):
                        subtr_lb = len(mask)-rb
                        lb = lb + subtr_lb
                        rb=len(mask)

                ind_arr = np.arange(lb,rb)
                mask[ind_arr]=True

                train_lko, test_lko, train_dict_lko = train_test_split(df, proxy_name, clim_name,
                                                          years_mask=mask,
                                                          train_age_std_coef=1)

                test_p_train_lko = pd.concat([test_lko,train_lko])
                
                proxy_lim = [min(test_p_train_lko[proxy_name]), 
                               max(test_p_train_lko[proxy_name])]
                age_lim = [min(test_p_train_lko[age_name]), 
                             max(test_p_train_lko[age_name])]


                surface = make_surface(train_lko, train_dict_lko, sm=sm[j],
                                       n_sq=n_sq[i],
                                       n = n,
                                       proxy_lim = proxy_lim,
                                       age_lim = age_lim,
                                       kernel=kernel,
                                       proxy_name=proxy_name,
                                       clim_var = clim_name,
                                       use_squares=use_squares,
                                       use_std=use_std,
                                       square_plot=False)

                pred_df = predict_on_surface(surface, train_dict_lko, train_lko, 
                                             test_lko,
                                             use_std=use_std,
                                             proxy_name=proxy_name,
                                             clim_var = clim_name,
                                             proxy_lim = proxy_lim,
                                             age_lim = age_lim,
                                             uncertainty_data_rep = None,
                                             uncertainty_instrumental = None,
                                             Z_shift=Z_shift)

                ypr = pred_df.loc[year]['preds']#????clim_name or std
                y_clim = test_lko[test_lko['years']==year][clim_name].values[0]

                preds.append(ypr)
                origs.append(y_clim)

              preds = np.array(preds)
              origs = np.array(origs)


              metrics_CE[i,j] = CE(origs,preds)
              metrics_corr[i,j] = np.corrcoef(origs, preds)[0][1]





      metrics = [metrics_corr, metrics_CE]
      names = ['Correlation','CE']
      #heatmap for each metrics
      for met in range(len(metrics)):
        fig, axs = plt.subplots(figsize=(len(sm), len(n_sq)))
        axs = sns.heatmap(metrics[met],
                              cmap="jet", annot=np.round(metrics[met],2),
                              annot_kws={'fontsize': 9},fmt = '',
                              xticklabels = np.round(sm,1),
                              yticklabels = np.round(n_sq,1))
        plt.xlabel('smooth')
        plt.ylabel('n squares')
        axs.set_title(names[met], fontsize = 15)
        plt.show()

      return metrics
