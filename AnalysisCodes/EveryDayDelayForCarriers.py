from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import numpy as np

s = pd.read_csv("G:\BD_Project_Data\plots_data\plotdelayseveryday.csv")
ps = s.pivot(index='DayOfMonth', columns='UniqueCarrier', values='avgDelay')[["\"AA\"","\"UA\"","\"AS\""]]

rcParams['figure.figsize'] = (8,5)
ps.plot(kind='bar', colormap='prism');
plt.xlabel('Day Of Month')
plt.ylabel('Average delay')
plt.title('Delay for each carrier everyday')
plt.show()