from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import numpy as np

df_airport_rout = pd.read_csv('G:\BD_Project_Data\plots_data\df_airport_rout_data16_123.csv')
df_airports = pd.read_csv('G:/BD_Project_Data/plots_data/airports_data.csv')

rcParams['figure.figsize'] = (14,10)

maping = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=-130, llcrnrlat=22, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=-60, urcrnrlat=50) #max longitude (urcrnrlon) and latitude (urcrnrlat)

maping.drawcoastlines()
maping.drawcountries()
maping.drawmapboundary()
maping.fillcontinents(color = 'white', alpha = 0.3)
##Run Twice
maping.shadedrelief()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def z_score(x):
    return (x-np.average(x))/np.std(x)
	
colorings = plt.get_cmap('afmhot')(np.linspace(0.0, 1.0, 30))
colorings=np.flipud(colorings)

#----- Scatter -------
range=max(df_airports['conFlight'])-min(df_airports['conFlight'])
al=np.array([sigmoid(x) for x in z_score(df_airports['delay'])])
xs,ys = maping(np.asarray(df_airports['lng']), np.asarray(df_airports['lat']))
val=df_airports['conFlight']*4000.0/range
##Run Twice
maping.scatter(xs, ys,  marker='o', s= val, alpha = 0.8,color=colorings[(al*20).astype(int)])

#----- Text -------
## Value to be changed 5000 in next line to be made around 60000 for one year
text_data=df_airports[(df_airports['conFlight']>5000) & (df_airports['IATA'] != 'HNL')]
xt,yt = maping(np.asarray(text_data['lng']), np.asarray(text_data['lat']))
txt=np.asarray(text_data['IATA'])
zp=zip(xt,yt,txt)


maping = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=-130, llcrnrlat=22, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=-60, urcrnrlat=50) #max longitude (urcrnrlon) and latitude (urcrnrlat)

maping.drawcoastlines()
maping.drawcountries()
maping.drawmapboundary()
maping.fillcontinents(color = 'white', alpha = 0.3)
maping.shadedrelief()

delay=np.array([sigmoid(x) for x in z_score(df_airports["delay"])])
##Value to be changed later depending on the error this is for one year
colorings = plt.get_cmap('afmhot')(np.linspace(0.0, 1.0, 40))
colorings=np.flipud(colorings)
xs,ys = maping(np.asarray(df_airports['lng']), np.asarray(df_airports['lat']))
xo,yo = maping(np.asarray(df_airport_rout['lng_x']), np.asarray(df_airport_rout['lat_x']))
xd,yd = maping(np.asarray(df_airport_rout['lng_y']), np.asarray(df_airport_rout['lat_y']))

maping.scatter(xs, ys,  marker='o',  alpha = 0.8,color=colorings[(delay*20).astype(int)])


al=np.array([sigmoid(x) for x in z_score(df_airport_rout["avgDelay"])])
f=zip(xo,yo,xd,yd,df_airport_rout['avgDelay'],al)
for row in f:
    plt.plot([row[0],row[2]], [row[1],row[3]],'-',alpha=0.07, \
             color=colorings[(row[5]*30).astype(int)] )
    

for row in zp:
    plt.text(row[0],row[1],row[2], fontsize=10, color='blue',)

print("Each line represents a route from the Origin to Destination airport.")
print("The redder line, the higher probablity of delay.")
    
plt.show()