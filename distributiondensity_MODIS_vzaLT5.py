import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import pandas as pd
import glob,matplotlib

file = ''
df = pd.read_csv(file)
lat = df_.loc[:,'lat']
lon = df_.loc[:,'lon']
xy = np.vstack([lat,lon])
weights = np.ones_like(lon) / len(lon)
z = gaussian_kde(xy)(xy)

fig=plt.figure(figsize=(2.8,3.7))
ax1=plt.axes([0.06,0.44,0.71,0.55])
b_map = Basemap(resolution='f', area_thresh=100, projection='cyl',\
                llcrnrlon = 104, llcrnrlat = 6, urcrnrlon = 125, urcrnrlat = 25,ax=ax1)   # 实例basemap对象125.614415,25.601117104.402332,6.031072
matplotlib.rcParams["font.family"] = "Times New Roman"  #全局Times new roman
csfont = {'fontname':'Times New Roman','fontsize':10}
parallels = [10,14,18,22]
b_map.drawparallels(parallels,labels=[1,0,0,0],rotation='vertical',**csfont)
meridians = [107,112,117,122]
#以10度为间隔画出西经180度到本初子午线经线， 并且在图像下侧设置经线标签
b_map.drawmeridians(meridians,labels=[0,0,1,0],**csfont)
b_map.drawcoastlines(linewidth=0.1)
b_map.fillcontinents(color='black')
#b_map.drawcountries()

jet = plt.get_cmap('jet') 
lonx, laty = b_map(lon, lat)
csOWL=b_map.scatter(lonx, laty,c=z, marker = 'o',s=0.1,cmap=jet,label='Field sites')


#ax.legend(handles=[cs],bbox_to_anchor=(0.43,0.25), borderaxespad=0.,fontsize=8)
cbar = b_map.colorbar(csOWL,location='right',pad="5%")
#csbb700OWLbar=b_map.colorbar(csbb700OWL,location='right',pad="5%")
cbar.set_label('Coverage probaility density',**csfont)

ax2=plt.axes([0.01,0.1,0.85,0.37])
ax2.yaxis.tick_right()
weights = np.ones_like(lon) / len(lon)
ax2.hist(lon, bins=100, color='blue',weights=weights)
ax2.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)
ax2.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
ax2.set_xlabel(u'Longitude(degree)', fontdict=csfont)
ax2.set_ylabel(u'Coverage probability', fontdict=csfont)
ax2.yaxis.set_label_coords(1.15, 0.5)
figname=r'C:\Users\lwk15\Downloads/test_300dpi.tif'  
plt.savefig(figname,dpi=300)
plt.show()
plt.close()