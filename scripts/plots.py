import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot3d(df, Z, proxy_lim, age_lim,type_p = ['scatter', 'wireframe'], name_='',
              clim_name='avg summer temperature', proxy_name='proxy',
              elev=None, azim=150):

    n = Z.shape[0] #number of values along the axis

    #axes for plot functions
    x_grid0 = np.linspace(proxy_lim[0],proxy_lim[1],n)
    y_grid0 = np.linspace(age_lim[0],age_lim[1],n)
    B1, B2 = np.meshgrid(x_grid0, y_grid0, indexing='xy')

    #plots with elev specification and without
    if elev is not None:
      fig, axs = plt.subplots(ncols=2,subplot_kw=dict(projection='3d', azim=azim,elev=elev),
                              gridspec_kw={'width_ratios': [100, 1]},
                              figsize=(30,20))
    else:
      fig, axs = plt.subplots(ncols=2,subplot_kw=dict(projection='3d', azim=azim),
                              gridspec_kw={'width_ratios': [100, 1]},
                              figsize=(30,20))
      
    ax = axs[0]
    axs[1].axis('off')

    ax.set_title(name_, fontdict={'fontsize': 30, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    if 'wireframe' in type_p:
      ax.plot_wireframe(B1, B2, Z)
    if 'surface' in type_p:
      ax.plot_surface(B1, B2, Z, alpha=0.6)
    if 'scatter' in type_p:
      ax.scatter3D(df[proxy_name], df['age'], df[clim_name], c='r', s=4)

    ax.set_xlabel(proxy_name,fontsize=20)
    ax.set_ylabel('age',fontsize=20)
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(clim_name,fontsize=20,rotation=90)
    plt.subplots_adjust(right=0.5)
    plt.show()



def plot2d(df, Z, name_='', clim_name='avg summer temperature',
           proxy_name='proxy'):
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
      proxy_name : str, optional
          name of the column in df that contains proxy values, default is 'proxy'
    """

    n = Z.shape[0] #number of values along the axis
    #axes for plot functions
    x_grid0 = np.linspace(min(df[proxy_name]), max(df[proxy_name])+5, n)
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
    proxy_rescale = []
    age_rescale = []
    for v in range(len(df[proxy_name])):
      # rescaling observed values for simultaneous display with Z grid
      proxy_rescale0 = n * np.abs(df[proxy_name].values[v]-
                                 min(x_grid0))/ np.abs(max(x_grid0)-min(x_grid0))
      age_rescale0 = n * np.abs(df['age'].values[v]-
                                min(y_grid0))/ np.abs(max(y_grid0)-min(y_grid0))
      proxy_rescale.append(proxy_rescale0)
      age_rescale.append(age_rescale0)

    #horizontal colorbar
    cm = plt.cm.get_cmap('YlOrBr')
    im = axs[0][0].scatter([proxy_rescale], [age_rescale],
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
    axs[0][0].set_xlabel(proxy_name,  fontsize=20)
    #vertical colorbar
    cbar1 = fig.colorbar(im0, cax=axs[0][1], orientation='vertical')
    cbar1.set_label('approximated values', fontsize=20)
    fig.suptitle(name_, fontsize=20)
    plt.show()



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
  plt.figure(figsize=(8, 6))
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
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=15)
  plt.show()


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
  plt.xlabel('years',fontsize=15)
  plt.ylabel(clim_var, fontsize=15)
  plt.legend(loc="lower right", prop={'size': 15})
  plt.show()
