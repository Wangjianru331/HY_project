import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
# from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.dates as mdates
import datetime


def draw(df1, df2):
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 6
            }
    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 7
             }
    plt.rc('font', family='Times New Roman', size=7)
    fig = plt.figure(11, figsize=(2.8, 1.8))

    band = pd.DataFrame([412, 443, 490, 520, 565, 670, 750, 865]).T
    band.columns = ['Rrs412_x', 'Rrs443_x', 'Rrs490_x', 'Rrs520_x', 'Rrs565_x', 'Rrs670_x', 'Rrs750_x', 'Rrs865_x']
    band1 = pd.DataFrame([412, 443, 490, 520, 565, 670, 750, 865]).T
    band1.columns = ['Rrs412_y', 'Rrs443_y', 'Rrs490_y', 'Rrs520_y', 'Rrs565_y', 'Rrs670_y', 'Rrs750_y', 'Rrs865_y']
    ax = plt.axes([0.1, 0.14, 0.89, 0.85])

    x_hy = df1['Time']
    x_mod = df2['Time']
    bands = ['HY-Band1', 'HY-Band2', 'HY-Band3', 'HY-Band4', 'HY-Band5', 'HY-Band6', 'HY-Band7', 'HY-Band8']
    bands_mod = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band10', 'Band11']
    bands_hy_label = ['412nm', '443nm', '490nm', '520nm', '565nm', '670nm', '750nm', '865nm']
    bands_mod_label = ['412nm', '443nm', '488nm', '531nm', '551nm', '667nm', '748nm', '869nm']
    for j, band in enumerate(bands):
        hy = df1[bands[j]]
        modis_ = df2[['Time', bands_mod[j]]].dropna(axis=0, how='any')

        modis_a = modis_[modis_['Time'] < datetime.datetime(2006, 6, 6)]
        x_mod_a = modis_a['Time']
        modisa = modis_a[bands_mod[j]]
        modis_b = modis_[modis_['Time'] > datetime.datetime(2006, 6, 6)]
        x_mod_b = modis_b['Time']
        modisb = modis_b[bands_mod[j]]

        #         x1 = np.array(g1.iloc[0, :])
        #         y1 = np.array(g1.iloc[1, :])

        #         if g1.shape[1] == 0:
        #             continue

        f2, = ax.plot(x_hy, hy, color='red', linewidth=0.3, linestyle='-')
        f1, = ax.plot(x_mod_a, modisa, color='blue', linewidth=0.3, linestyle='-')
        f1, = ax.plot(x_mod_b, modisb, color='blue', linewidth=0.3, linestyle='-')
        ax.set_xlabel(u'Date', fontdict=font1)
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)

        ax.set_ylabel(u'Mean $nRrs$ at TOA $(sr^{-1})}$', fontdict=font1)
        ax.yaxis.set_label_coords(-0.065, 0.5)

        ax.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)

        #         xlabels = np.array([2002])
        #         ymin, ymax = ax.get_ylim()
        #         ylabels = np.append(np.arange(0, ymax, 0.01)[0], np.arange(0, ymax, 0.01))
        #         ylabels = np.array([0, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
        #
        #         labels = ax.get_xticklabels()
        #         [label for label in labels]
        #         print(labels)
        #         ax.set_xticklabels([label for label in labels], fontdict=font1)
        #         ax.set_yticklabels(ylabels, fontdict=font1)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.legend((f2, f1), ['HY1B COCTS ' + bands_hy_label[j], 'Terra MODIS ' + bands_mod_label[j]], loc='upper right',
                  fontsize=6)

        figname = r'G:\hyProject/nRrsAtTOA_cocts-modis_' + bands[j] + '_300dpi.png'
        plt.savefig(figname, dpi=300)
        # plt.savefig(figname[0:-10] + '600dpi.png', dpi=600)
        plt.show()
    plt.close()


if __name__ == '__main__':
    data1 = pd.read_excel('G:\hyProject/HY-MODIS-timeseriesResult.xlsx', sheetname='HY-term', header=0)
    data2 = pd.read_excel('G:\hyProject/HY-MODIS-timeseriesResult.xlsx', sheetname='MOD 0-0.2', header=0)

    draw(data1, data2)