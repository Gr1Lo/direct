library(dplR)
library(utils)
#install.packages("remotes")
#remotes::install_github("konradmayer/trlboku")
library(trlboku)

#reading rwl and pth-file
synth_rwl = read.rwl('synthetic_ds.rwl',
                     format="tucson")
synth_pth = read.table('synthetic_ds.pth',
                       header = TRUE)

#rcs and rcs-sf chronologies
synth_rwi <- rcs(rwl = synth_rwl, po = synth_pth, biweight = TRUE)
synth_sf = sf_rcs(synth_rwl, synth_pth)
synth_rwi_sf = synth_sf$rwi
rwl_crn <- chron(synth_rwi)
rwl_crn_sf <- chron(synth_rwi_sf)

#merging rcs and rcs-sf chronologies
rwl_crn <- cbind(years = rownames(rwl_crn), rwl_crn)
rwl_crn_sf <- cbind(years = rownames(rwl_crn_sf), rwl_crn_sf)
xy <- merge(rwl_crn, rwl_crn_sf, by.x = 'years', by.y = 'years')
colnames(xy)[2] ="rcs_crn"
colnames(xy)[4] ="rcs_sf_crn"

rwis = xy[c('years',"rcs_crn","rcs_sf_crn")]
write.csv(rwis, 'rwis.csv', row.names=FALSE)

