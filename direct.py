import os
import sys
import pandas as pd
import numpy as np
import re
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import imageio
import gc
from scipy.spatial import distance




def plot3d_2d(df, Z, n=500, type_p = ['scatter', 'wireframe'], gif = False):
    vals0 = df['vals'].astype(float).to_numpy()
    age0 = df['age'].astype(float).to_numpy()
    mean0 = df['mean'].astype(float).to_numpy()
    x_grid0 = np.linspace(min(vals0), max(vals0), n)
    y_grid0 = np.linspace(min(age0), max(age0), n)
    B1, B2 = np.meshgrid(x_grid0, y_grid0, indexing='xy')

    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    if 'wireframe' in type_p:
      ax.plot_wireframe(B1, B2, Z)
    if 'surface' in type_p:
      ax.plot_surface(B1, B2, Z, alpha=0.6)
    if 'scatter' in type_p:
      ax.scatter3D(vals0, age0, mean0, c='r', s=2)

    ax.view_init(elev=None, azim=150)
    plt.xlabel("vals")
    plt.ylabel("age")
    plt.show()


    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(Z)
    xi = list(range(len(x_grid0)))
    yi = list(range(len(y_grid0)))
    plt.yticks(yi[0::100],np.round(y_grid0,2)[0::100])
    plt.xticks(xi[0::100],np.round(x_grid0,1)[0::100])
    plt.xlabel("vals")
    plt.ylabel("age")
    plt.colorbar()
    plt.show()

    if gif == True:
      for ii in range(0,360,10):
              print(ii)
              ax.view_init(elev=None, azim=ii)
              plt.savefig("drive/MyDrive/direct/plots/%d.png" % ii)

      # Build GIF
      frames = []
      lst = os.listdir('drive/MyDrive/direct/plots')
      print(sorted(lst))
      for filename in sorted(lst):
              image = imageio.imread('drive/MyDrive/direct/plots/' + filename)
              frames.append(image)
      imageio.mimsave('drive/MyDrive/direct/direct_gif7.gif', frames, format='GIF', duration=1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx






# function for reading files in rwl format
def process_rwl_pandas(filename, no_data, ind_length = 8, filename_pth=None):
      '''
      changed from https://github.com/OpenDendro/dplPy/blob/main/src/readers.py
      '''

      print("\nAttempting to read input file: " + os.path.basename(filename)
              + " as .rwl format\n")
      
      if filename_pth is not None:
        f_age = []
        with open(filename_pth, "r") as pth_file:
            lines = pth_file.readlines()
            if len(lines[0].split())==1:
                t_val = 0
            else:
                t_val = 1
            for line in lines:
              line0 = line.split()
              if t_val == 0:
                f_age.append(int(line))
              else:
                f_age.append(int(line0[1]))

      with open(filename, "r") as rwl_file:
          lines = rwl_file.readlines()
          rwl_data = {}
          first_date = sys.maxsize
          last_date = 0

          # read through lines of file and store raw data in a dictionary
          for line in lines:
              if ind_length is None:
                line = re.sub("[A-Za-z]+", lambda ele:  ele[0] + " ", line)
                line = line.rstrip("\n").split()
              else:
                line = line[:ind_length] + ' ' + line[ind_length:]
                line = line.rstrip("\n").split()

              ids = line[0]
              date = int(line[1])

              if ids not in rwl_data:
                  rwl_data[ids] = [date, []]

              # keep track of the first and last date in the series
              if date < first_date:
                  first_date = date
              if (date + len(line) - 3) > last_date:
                  last_date = date + len(line) - 3
              
              for i in range(2, len(line)):
                  try:
                      data = float(int(line[i]))/100
                      if data == no_data/100:
                          continue
                  except ValueError:
                      data = np.nan
                  rwl_data[ids][1].append(data)

      # create an array of indexes for the dataframe
      indexes = []
      for i in range(first_date, last_date):
          indexes.append(i)

      # create a new dictionary to store the data in a way more suited for the
      # dataframe
      '''refined_data = {}
      for key, val in rwl_data.items():
          front_addition = [np.nan] * (val[0]-first_date)
          end_addition = [np.nan] * (last_date - (val[0] + len(val[1])))
          refined_data[key] = front_addition + val[1] + end_addition

      df = pd.DataFrame.from_dict(refined_data)
      df.insert(0, "Year", indexes)
      df.set_index("Year")'''

      df = pd.DataFrame(columns=['years', 'vals', 'age', 'file'])
      cou = 0
      for key, val in rwl_data.items():
        years = []
        files = []
        
        if filename_pth is not None:
            if t_val == 0:
                age = f_age[cou]
            else:
                age = val[0] - f_age[cou]
                print(val[0])
                print(age)
        else:
          age = 0
        cou += 1
        ages = []
        for i in range(val[0], val[0] + len(val[1])):
            years.append(i)
            files.append(key)
            ages.append(age)
            age += 1
            
        m_array = np.array([years, val[1], ages, files])
        df1 = pd.DataFrame(m_array.T, columns=df.columns)
        df = df.append(df1,ignore_index=True)

      df['years']=df['years'].astype(int)
      return df


def read_meteo(file_p, sep = '\t', names=list(range(0,12)), months=[5,6,7]):
    df_meteo = pd.read_csv(file_p, sep=sep, names=names)
    df_meteo = df_meteo[months]
    df_meteo['mean'] = df_meteo.mean(axis=1)
    df_meteo['years'] = df_meteo.index
    return df_meteo[['years', 'mean']]

class direct:
  def __init__(self, rwl, meteo, no_data=-9999, filename_pth=None):
    df1 = process_rwl_pandas(rwl, no_data=no_data, filename_pth=filename_pth)
    df_meteo = read_meteo(meteo)
    result = pd.merge(df1, df_meteo, on="years", how='left')
    result = result[~result['mean'].isna()]

    ########## Standardizing data
    result['vals'] = result['vals'].astype(float).to_numpy()
    result['age'] = result['age'].astype(float).to_numpy()
    result['mean'] = result['mean'].astype(float).to_numpy()

    self.vals_std = result['vals'].std()
    self.vals_mean = result['vals'].mean()
    self.age_std = result['age'].std()
    self.age_mean = result['age'].mean()
    result['vals'] = (result['vals'] - self.vals_mean) / self.vals_std
    result['age'] = (result['age'] - self.age_mean) / self.age_std
    #result['mean'] = (result['mean'] - result['mean'].mean()) / result['mean'].std()
    #########

    self.min_vals = min(result['vals'])
    self.max_vals = max(result['vals'])
    self.min_age = min(result['age'])
    self.max_age = max(result['age'])

    self.merged_df = result

  def train_test_split(self, r):
    df = self.merged_df
    unique_list = np.unique(df['file'])
    nr = int(r * len(unique_list))
    sel_n = np.random.choice(unique_list, nr, replace=False)
    self.test = df[df['file'].isin(sel_n)]
    self.train = df[~df['file'].isin(sel_n)]

  def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

  def interpolate_SBS(self, n):
      '''
      #IQR
      Q1 = result['mean'].quantile(0.25)
      Q3 = result['mean'].quantile(0.75)
      IQR = Q3 - Q1    

      filter = (result['mean'] >= Q1 - 1.5 * IQR) & (result['mean'] <= Q3 + 1.5 *IQR)
      result = result.loc[filter]'''

      vals = self.train['vals'].astype(float).to_numpy()
      age = self.train['age'].astype(float).to_numpy()
      vals0 = self.merged_df['vals'].astype(float).to_numpy()
      age0 = self.merged_df['age'].astype(float).to_numpy()
      mean = self.train['mean'].astype(float).to_numpy()

      x_grid0 = np.linspace(min(vals0), max(vals0), n)
      y_grid0 = np.linspace(min(age0), max(age0), n)
      
      #interpolation
      m = sp.interpolate.SmoothBivariateSpline(age, vals, mean, s =50000,
                                                  kx=2, ky=2)

      Z = m.__call__(x_grid0, y_grid0)
      Z = np.where(Z > max(mean),np.nan,Z)
      Z = np.where(Z < min(mean),np.nan,Z)

      return Z


  def mean_in_circle(self, n, norm_radius):
      Z = np.empty([n, n])
      Z[:] = np.nan
      mdf = self.train.copy()
      vals0 = self.merged_df['vals'].to_numpy()
      age0 = self.merged_df['age'].to_numpy()
      mean = mdf['mean'].to_numpy()

      x_grid0 = np.linspace(min(vals0), max(vals0), n)
      y_grid0 = np.linspace(min(age0), max(age0), n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')
      
      Z_list = []
      xcou = 0
      for xi in x_grid0:
        ycou = 0
        #print(xi)
        mdf_x = mdf[(mdf['age'] > xi-norm_radius) & (mdf['age'] < xi+ norm_radius)]
        Z_list_x = []
        for yi in y_grid0:
          mdf_xy = mdf_x[(mdf_x['vals'] > yi - norm_radius) & (mdf_x['vals'] < yi + norm_radius)]
          if len(mdf_xy) != 0:
            nodes = mdf_xy[['vals', 'age']].to_numpy()
            dist = distance.cdist(nodes, [[yi,xi]])
            mdf_xy = mdf_xy[dist < norm_radius]
            if len(mdf_xy) > 2:
              mean_mean = mdf_xy["mean"].mean()
              Z[xcou, ycou] = mean_mean

          ycou = ycou + 1
        xcou = xcou + 1

      mask = np.where(~np.isnan(Z))
      #interp = sp.interpolate.NearestNDInterpolator(np.transpose(mask), Z[mask])
      interp = sp.interpolate.LinearNDInterpolator(np.transpose(mask), Z[mask])
      Z = interp(*np.indices(Z.shape))
      Z = np.where(Z > max(mean),np.nan,Z)
      Z = np.where(Z < min(mean),np.nan,Z)
      
      return Z

  def iter_method(self, m, n):
    all_Z = np.empty([m, n, n])
    all_Z[:] = np.nan
    df = self.train
    vals0 = self.merged_df['vals'].to_numpy()
    age0 = self.merged_df['age'].to_numpy()
    for ii in range(m):
      #print(ii)
      unique_list = np.unique(df['file'])
      sel_n = np.random.choice(unique_list, 10, replace=False)
      result = df[df['file'].isin(sel_n)]

      vals = result['vals'].to_numpy()
      age = result['age'].to_numpy()
      mean = result['mean'].to_numpy()

      m = sp.interpolate.SmoothBivariateSpline(age, vals, mean,
                                                kx=1, ky=1, s=10000)
                                                #w = 1/dists)#,
                                                #eps = 0.00001)
      x_grid0 = np.linspace(min(vals0), max(vals0), n)
      y_grid0 = np.linspace(min(age0), max(age0), n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')
    
      mask_x = (x_grid0<max(vals)) & (x_grid0>min(vals))
      x_ind = np.array([i for i, x in enumerate(mask_x) if x])
      mask_y = (y_grid0<max(age)) & (y_grid0>min(age))
      y_ind = np.array([i for i, x in enumerate(mask_y) if x])
      x_grid = x_grid0[mask_x]
      y_grid = y_grid0[mask_y]
      B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

      Z = np.round(m.__call__(x_grid, y_grid),2)
      Z = np.where(Z > max(mean),max(mean),Z)
      Z = np.where(Z < min(mean),min(mean),Z)
      Z0 = np.zeros((len(x_grid0),len(y_grid0)))
      Z0 = np.full((len(x_grid0),len(y_grid0)),np.nan)
      Z0[min(x_ind):max(x_ind)+1,min(y_ind):max(y_ind)+1] = Z
      #Z0 = Z0.T

      all_Z[ii] = Z0
      
      '''fig = plt.figure(figsize=(10,6))
      ax = Axes3D(fig)
      ax.xaxis.set_major_locator(plt.MaxNLocator(8))
      ax.yaxis.set_major_locator(plt.MaxNLocator(8))

      ax.plot_wireframe(B1_0, B2_0, Z0)
      #ax.plot_surface(B1, B2, Z0, alpha=0.6)
      ax.scatter3D(vals, age, mean, c='r', s=10)
      ax.scatter3D(xr, yr, zr, c='b', s=1)
      ax.view_init(elev=None, azim=150)

      plt.xlabel("vals")
      plt.ylabel("age")
      plt.show()

      fig, ax = plt.subplots(figsize=(10,10))
      plt.imshow(Z)
      xi = list(range(len(x_grid0)))
      yi = list(range(len(y_grid0)))
      #??????????
      plt.yticks(yi[0::100],np.round(y_grid0,2)[0::100])
      plt.xticks(xi[0::100],np.round(x_grid0,1)[0::100])
      plt.xlabel("vals")
      plt.ylabel("age")
      plt.colorbar()
      plt.show()'''


      del Z0
      del Z
      del m
      gc.collect()

    gc.collect()
    mean_Z = np.empty([n, n])
    for i in range(all_Z.shape[1]):
        for j in range(all_Z.shape[2]):
            mean_Z[i,j] = np.nanmean(all_Z[:,i,j])

    gc.collect()

    return mean_Z

  def m_sq_method(self, m, n, n_sq):
    all_Z = np.empty([m, n, n])
    all_Z[:] = np.nan
    df = self.train
    merged_df = self.merged_df
    vals0 = merged_df['vals'].to_numpy()
    age0 = merged_df['age'].to_numpy()
    for ii in range(m):
      #print(ii)
      unique_list = np.unique(df['file'])
      sel_n = np.random.choice(unique_list, 10, replace=False)
      result = df[df['file'].isin(sel_n)]

      vals = result['vals'].to_numpy()
      age = result['age'].to_numpy()
      mean = result['mean'].to_numpy()

      Z = np.empty([n_sq, n_sq])
      Z[:] = np.nan

      x_grid = np.linspace(min(vals), max(vals), n)
      y_grid = np.linspace(min(age), max(age), n)
      B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

      x_grid_sq = np.linspace(min(vals), max(vals), n_sq)
      y_grid_sq = np.linspace(min(age), max(age), n_sq)
      B1_sq, B2_sq = np.meshgrid(x_grid_sq, y_grid_sq, indexing='xy')

      xcou = 0
      st = x_grid_sq[1]-x_grid_sq[0]
      for xi in x_grid_sq[:-1]:
        ycou = 0
        mdf_x = result[(result['age'] > xi-0.5*st) & (result['age'] < xi+0.5*st)]
        for yi in y_grid_sq[:-1]:
          mdf_xy = result[(result['vals'] > yi-0.5*st) & (result['vals'] < yi+0.5*st)]
          if len(mdf_xy) != 0:
              mean_mean = mdf_xy["mean"].mean()
              Z[xcou, ycou] = mean_mean

          ycou = ycou + 1
        xcou = xcou + 1

      zr = Z.ravel()[~np.isnan(Z.ravel())]
      xr = B1_sq.ravel()[~np.isnan(Z.ravel())]
      yr = B2_sq.ravel()[~np.isnan(Z.ravel())]

      m_sq = sp.interpolate.SmoothBivariateSpline(yr, xr, zr, kx=3, ky=3)

      x_grid0 = np.linspace(min(vals0), max(vals0), n)
      y_grid0 = np.linspace(min(age0), max(age0), n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')
      mask_x = (x_grid0<max(vals)) & (x_grid0>min(vals))
      x_ind = np.array([i for i, x in enumerate(mask_x) if x])
      mask_y = (y_grid0<max(age)) & (y_grid0>min(age))
      y_ind = np.array([i for i, x in enumerate(mask_y) if x])
      x_grid = x_grid0[mask_x]
      y_grid = y_grid0[mask_y]

      Z = np.round(m_sq.__call__(x_grid, y_grid),2)
      Z = np.where(Z > max(mean),max(mean),Z)
      Z = np.where(Z < min(mean),min(mean),Z)
      Z0 = np.zeros((len(x_grid0),len(y_grid0)))
      Z0 = np.full((len(x_grid0),len(y_grid0)),np.nan)
      Z0[min(x_ind):max(x_ind)+1,min(y_ind):max(y_ind)+1] = Z
      #Z0 = Z0.T

      all_Z[ii] = Z0
      

      '''fig = plt.figure(figsize=(10,6))
      ax = Axes3D(fig)
      ax.xaxis.set_major_locator(plt.MaxNLocator(8))
      ax.yaxis.set_major_locator(plt.MaxNLocator(8))

      ax.plot_wireframe(B1_0, B2_0, Z0)
      #ax.plot_surface(B1, B2, Z0, alpha=0.6)
      ax.scatter3D(vals, age, mean, c='r', s=10)
      ax.scatter3D(xr, yr, zr, c='b', s=1)
      ax.view_init(elev=None, azim=150)

      plt.xlabel("vals")
      plt.ylabel("age")
      plt.show()

      fig, ax = plt.subplots(figsize=(10,10))
      plt.imshow(Z)
      xi = list(range(len(x_grid0)))
      yi = list(range(len(y_grid0)))
      #??????????
      plt.yticks(yi[0::100],np.round(y_grid0,2)[0::100])
      plt.xticks(xi[0::100],np.round(x_grid0,1)[0::100])
      plt.xlabel("vals")
      plt.ylabel("age")
      plt.colorbar()
      plt.show()'''


      del Z0
      del Z
      del m_sq
      del mdf_x
      del mdf_xy
      gc.collect()

    gc.collect()
    mean_Z = np.empty([n, n])
    for i in range(all_Z.shape[1]):
        for j in range(all_Z.shape[2]):
            mean_Z[i,j] = np.nanmean(all_Z[:,i,j])

    gc.collect()

    return mean_Z

  def sq_method(self, n, n_sq):
      Z = np.empty([n_sq, n_sq])
      Z[:] = np.nan
      mdf = self.train.copy()
      vals0 = self.merged_df['vals'].to_numpy()
      age0 = self.merged_df['age'].to_numpy()
      mean = mdf['mean'].to_numpy()

      x_grid0 = np.linspace(min(vals0), max(vals0), n)
      y_grid0 = np.linspace(min(age0), max(age0), n)
      B1_0, B2_0 = np.meshgrid(x_grid0, y_grid0, indexing='xy')

      x_grid = np.linspace(min(vals0), max(vals0), n_sq)
      y_grid = np.linspace(min(age0), max(age0), n_sq)
      B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
      
      Z_list = []
      xcou = 0

      st = x_grid[1]-x_grid[0]
      for xi in x_grid[:-1]:
        ycou = 0
        mdf_x = mdf[(mdf['age'] > xi-0.5*st) & (mdf['age'] < xi+0.5*st)]
        for yi in y_grid[:-1]:
          mdf_xy = mdf_x[(mdf_x['vals'] > yi-0.5*st) & (mdf_x['vals'] < yi+0.5*st)]
          if len(mdf_xy) != 0:
              mean_mean = mdf_xy["mean"].mean()
              Z[xcou, ycou] = mean_mean

          ycou = ycou + 1
        xcou = xcou + 1

      '''fig, ax = plt.subplots(figsize=(10,10))
      plt.imshow(Z)
      xi = list(range(len(x_grid)))
      yi = list(range(len(y_grid)))
      #??????????
      plt.yticks(yi[0::100],np.round(y_grid,2)[0::100])
      plt.xticks(xi[0::100],np.round(x_grid,1)[0::100])
      plt.xlabel("vals")
      plt.ylabel("age")
      plt.colorbar()
      plt.show()
      '''

      zr = Z.ravel()[~np.isnan(Z.ravel())]
      xr = B1.ravel()[~np.isnan(Z.ravel())]
      yr = B2.ravel()[~np.isnan(Z.ravel())]

      m = sp.interpolate.SmoothBivariateSpline(xr, yr, zr, kx=3, ky=3)
      
      Z = m.__call__(x_grid0, y_grid0).T

      Z = np.where(Z > max(mean),np.nan,Z)
      Z = np.where(Z < min(mean),np.nan,Z)
      
      return Z


  def predict_in_grid(self, df, grid, n):
    df0 = df.copy()
    vals0 = self.merged_df['vals'].astype(float).to_numpy()
    age0 = self.merged_df['age'].astype(float).to_numpy()
    x_grid0 = np.linspace(min(vals0), max(vals0), n)
    y_grid0 = np.linspace(min(age0), max(age0), n)
    preds = []

    ##############?????????????##############
    for index, row in df0.iterrows():
      '''x = find_nearest(x_grid0, row['vals'])
      y = find_nearest(y_grid0, row['age'])
      preds.append(grid[y, x])'''

      x = find_nearest(x_grid0, row['age'])
      y = find_nearest(y_grid0, row['vals'])
      preds.append(grid[x, y])
    ##############?????????????##############
    df0['preds'] = np.array(preds)
    return df0



  def direct_statistics(self, grid, n, name):
    #Calibration and verification statistics
    print(name)
    test = self.test
    train = self.train
    merged_df = self.merged_df
    pred_test = self.predict_in_grid(test, grid, n)[['mean','preds']].dropna()
    pred_train = self.predict_in_grid(train, grid, n)[['mean','preds']].dropna()
    pred_all = self.predict_in_grid(merged_df, grid, n)[['mean','preds']].dropna()

    print('Коэффициент корреляции на тестовой выборке: ',
          round(pred_test.corr(method='pearson')['preds'][0], 3))
    print('Коэффициент корреляции на тренировочной выборке: ',
          round(pred_train.corr(method='pearson')['preds'][0], 3))
    print('Коэффициент корреляции на всей выборке: ',
          round(pred_all.corr(method='pearson')['preds'][0], 3))

    RE = 1 - np.sum((pred_train['mean']-pred_train['preds'])**2) / np.sum((pred_train['mean']-np.mean(pred_train['mean']))**2)
    CE = 1 - np.sum((pred_test['mean']-pred_test['preds'])**2) / np.sum((pred_test['mean']-np.mean(pred_test['mean']))**2)

    print('RE: ', round(RE,3))
    print('CE: ', round(CE,3))

    RMSE_test = ((pred_test['mean'] - pred_test['preds']) ** 2).mean() ** .5
    RMSE_train = ((pred_train['mean'] - pred_train['preds']) ** 2).mean() ** .5
    RMSE_all = ((pred_all['mean'] - pred_all['preds']) ** 2).mean() ** .5

    print('RMSE для тестовой выборки: ', round(RMSE_test,3))
    print('RMSE для тренировочной выборки: ', round(RMSE_train,3))
    print('RMSE для всей выборки: ', round(RMSE_all,3))
