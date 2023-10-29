library(dplR)
library(readxl)
df =read.rwl('Cauc_BI_inv.rwl')

#select only D18* and D09S* samples
D18_D09S = df[ , (grepl( "D18" , names( df ) ) | grepl( "D09S" , names( df ) )) ]
D18_D09S = D18_D09S[rowSums(is.na(D18_D09S)) != length(colnames(D18_D09S)), ] 

#ssf-standartization
D18_D09S_SSF <- ssf(D18_D09S, method='AgeDepSpline', nyrs=50)
D18_D09S_SSF$year = rownames(D18_D09S_SSF)

#reading the meteo-data
meteo <- read_excel("BI rec.xlsx")

#merging proxy and meteo-data dataframes
m_df = merge(x = D18_D09S_SSF, y = meteo, by = "year", all.x = TRUE)

#scaling function for statistical modelling step
scaling <- function(df, train_year_begin, train_year_end, 
                    test_year_begin=0, test_year_end=0,
                    new_col = 'sfc_resc5180') {
  
  train_m_df = df[(df$year>=train_year_begin) & (df$year<=train_year_end),]
  
  sd_tri = sd(train_m_df[,'sfc'], na.rm = TRUE)
  to0mean = 0 - mean(train_m_df[,'sfc'], na.rm = TRUE)
  df[,new_col] = df[,'sfc']+to0mean
  sd_instr = sd(train_m_df[,'VI IX instr'], na.rm = TRUE)
  mean_instr = mean(train_m_df[,'VI IX instr'], na.rm = TRUE)
  
  df[,new_col] = df[,new_col]*(sd_instr/sd_tri) + mean_instr
  
  return(df)
}

#new column for calibration at 1951-1980
n_df = scaling(m_df,1951,1980,new_col = 'sfc_resc5180')
#new column for calibration on 1981-2011
n_df = scaling(n_df,1981,2011,new_col = 'sfc_resc8111')
#new column for calibration on 1951-2011
n_df = scaling(n_df,1951,2011,new_col = 'sfc_resc')

#save the results
write.csv(n_df,"BI rec.csv")
