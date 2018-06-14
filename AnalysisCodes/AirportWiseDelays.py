from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import numpy as np

airports_data = pd.read_csv('G:/BD_Project_Data/plots_data/airports_data.csv')

rcParams['figure.figsize'] = (14,10)


mappings = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=-130, llcrnrlat=22,
            urcrnrlon=-60, urcrnrlat=50)

mappings.drawcoastlines()
mappings.drawcountries()
mappings.drawmapboundary()
mappings.fillcontinents(color = 'white', alpha = 0.3)
##Run Twice
mappings.shadedrelief()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def z_score(x):
    return (x-np.average(x))/np.std(x)

# To create a color map
colorings = plt.get_cmap('afmhot')(np.linspace(0.0, 1.0, 30))
colorings = np.flipud(colorings)

#----- Scatter -------
range=max(airports_data['conFlight'])-min(airports_data['conFlight'])
al=np.array([sigmoid(x) for x in z_score(airports_data['delay'])])
xs,ys = mappings(np.asarray(airports_data['lng']), np.asarray(airports_data['lat']))
val=airports_data['conFlight']*4000.0/range
##Run Twice
mappings.scatter(xs, ys,  marker='o', s= val, alpha = 0.8,color=colorings[(al*20).astype(int)])

#----- Text -------
## Value to be changed 5000 in next line to be made around 60000 for one year
text_dataframe=airports_data[(airports_data['conFlight']>5000) & (airports_data['IATA'] != 'HNL')]
xt,yt = mappings(np.asarray(text_dataframe['lng']), np.asarray(text_dataframe['lat']))
txt=np.asarray(text_dataframe['IATA'])
zp=zip(xt,yt,txt)
for row in zp:
    #print zp[2]
    plt.text(row[0],row[1],row[2], fontsize=10, color='blue')


##Each marker is an airport.
##Size of markers: Airport Traffic (larger means higher number of flights in year)
##Color of markers: Average Flight Delay (Redder means longer delays)	

plt.show()