import pandas as pd
import numpy as np

def rwl2pandas(rwl_path, no_data_value=-9999, ind_length = 8, pth_path=None,
               proxy_name='proxy'):
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
  proxy_name : str, optional
      name for column of values in output dataframe, default is 'proxy'

  Returns
  -------
  df : pandas dataframe object
      Dataframe with columns:
          'years' - year of the observation,
          proxy_name - measurand,
          'age' - age of the tree at the observation,
          'file' - name of the series
          'log_'+proxy_name - log-transformed measurand
          'log_age' - log-transformed age
  """

  f_age = [] #list with first ages
  f_age_files = []
  if pth_path is not None:
        with open(pth_path) as file:
          while (line := file.readline().rstrip()):
            if len(line.split())>1:
                #for pth-file with multiple columns
                if 'series' not in line:
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
        f_age.append(1)

  all_l = []
  ind_file_list = []
  df = pd.DataFrame(columns=['years', proxy_name, 'age', 'file'])
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
              age_first = f_age[ind]#year_line-f_age[ind]                                  !!!!!!!!!!!!!!!

          ind_file_list.append(ind_file)
          cou += 1

        #appending every observation to the list of lists that further will be
        #inserted to output df
        for i in range(1, len(obs)):
            ob = float(obs[i])
            if ob != no_data_value:
              all_l.append([year_line, ob, age_first, ind_file, np.log(ob), np.log(age_first)])

            year_line += 1
            age_first += 1

  df = pd.DataFrame.from_records(all_l,columns=['years', proxy_name,
                                                'age', 'file', 'log_'+proxy_name, 'log_age'])
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
    if 'years' in df_meteo.columns:
        df_meteo.index = df_meteo.loc[:,'years']

    df_meteo = df_meteo.iloc[:, months]
    #creating a new column with averaged values
    if len(months)>1:
      df_meteo.loc[:,clim_name] = df_meteo.mean(axis=1)
    else:
      df_meteo.loc[:,clim_name] = df_meteo.iloc[:,[0]]

    df_meteo.loc[:,'years'] = df_meteo.index
    return df_meteo[['years', clim_name]]

def direct_read(rwl, meteo, no_data_value=-9999, ind_length=8, pth_path=None,
                clim_name='avg summer temperature', proxy_name='proxy',
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
    proxy_name : str, optional
        name for column of values in output dataframe, default is 'proxy'
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
            proxy_name - measurand,
            'age' - age of the tree at the observation,
            'file' - name of the series
            clim_name - averaged climatic value
    """
    #Reading rwl-files and inserting values to pandas dataframe
    df1 = rwl2pandas(rwl, no_data_value=no_data_value, pth_path=pth_path,
                             proxy_name=proxy_name)
    #Reading meteo-data
    df_meteo = read_meteo(meteo,sep=meteo_sep, clim_name=clim_name,
                          names=meteo_names, months=meteo_months)
    #Merging rwl and meteo-data by years, clim_name column could contains
    #NaN values
    result = pd.merge(df1, df_meteo, on="years", how='left')
    result.loc[:,proxy_name] = result[proxy_name].astype(float).to_numpy()
    result.loc[:,'age'] = result['age'].astype(float).to_numpy()
    result.loc[:,clim_name] = result[clim_name].astype(float).to_numpy()

    return result








def standardize_train(df, proxy_name,  clim_name, age_var='age', train_age_std_coef=1):
  train=df.copy()
  #Standardization parameters for proxy
  train_proxy_std = train[proxy_name].std()
  train_proxy_mean = train[proxy_name].mean()
  #Standardization parameters for age
  train_age_std = train[age_var].std()*train_age_std_coef
  train_age_mean = train[age_var].mean()
  #Standardization parameters for climatic data
  av_df = train[[clim_name,'years']].groupby(['years']).mean()
  train_mean_std = np.std(av_df[clim_name])
  train_mean_mean = np.mean(av_df[clim_name])

  train_dict = {'train_proxy_std':train_proxy_std,
                'train_proxy_mean':train_proxy_mean,
                'train_age_std':train_age_std,
                'train_age_mean':train_age_mean,
                'train_mean_std':train_mean_std,
                'train_mean_mean':train_mean_mean}


  train.loc[:,'std_'+proxy_name] = (train[proxy_name] -
                                     train_proxy_mean) / train_proxy_std
  train.loc[:,'std_'+age_var] = (train[age_var] - train_age_mean) / train_age_std
  train.loc[:,'std_'+clim_name] = (train[clim_name] -
                                     train_mean_mean) / train_mean_std

  return train,train_dict

def standardize_test(train_dict, df, proxy_name,  clim_name, age_var='age'):
  test=df.copy()
  test.loc[:,'std_'+proxy_name] = (test[proxy_name] -
                                    train_dict['train_proxy_mean']) / train_dict['train_proxy_std']
  test.loc[:,'std_'+age_var] = (test[age_var] - train_dict['train_age_mean']) / train_dict['train_age_std']
  test.loc[:,'std_'+clim_name] = (test[clim_name] -
                                    train_dict['train_mean_mean']) / train_dict['train_mean_std']

  return test

def train_test_split(df, proxy_name,  clim_name, age_var='age', r=None,
                     years_mask=None, test_mask=None,
                     train_age_std_coef=1):
    """Splits original dataframe into train and test dataframes

      Parameters
      ----------
      df : pandas dataframe object
          dataframe in form like returned by direct_read()
      proxy_name : str
          name of the column that contains proxy values
      clim_name : str
        name of the column in df that contains climatic values
      r : float, optional
          the proportion of the test sample,
          user should specify either r or years_mask
      years_mask : boolean list, optional
          boolean list for years, True for test sample,
          user should specify either r or years_mask
      test_mask : boolean list, optional
          boolean list for years, True for test sample,
          is used when user want to exclude some years from test
          or include some years from training
      train_age_std_coef : float, optional
          regulates the value of the standard deviation by which 
          the age will be normalized

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
      train = df[~df['years'].isin(sel_n)]
      test = df[df['years'].isin(sel_n)]
      if test_mask is not None:
          sel_n_t = [a for a, b in zip(all_years, test_mask) if b]
          test = df[~df['years'].isin(sel_n_t)]

    elif r is not None:
      #without respect to an order of observations
      unique_list = np.unique(df['years'])
      nr = int(r * len(unique_list))
      sel_n = np.random.choice(unique_list, nr, replace=False)
      test = df[df['years'].isin(sel_n)]
      train = df[~df['years'].isin(sel_n)]
    else:
      print('r or years_mask should be specified')

    train,train_dict = standardize_train(train, proxy_name,  clim_name, age_var,train_age_std_coef)
    test = standardize_test(train_dict, test, proxy_name,  clim_name, age_var)




    return train, test, train_dict
