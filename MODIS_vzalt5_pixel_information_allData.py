# 抽取MODIS观测天顶角小于5度的数据
# 输入为
# version： 1
import pandas as pd
import numpy as np
import datetime
import glob
import os



if __name__ == '__main__':
    files = glob.glob(r'F:\MODIS/MOD021KM*_pixel_level_info.csv')
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df = df.dropna(axis=0,how='any')
        df = df[df['vza']*0.01<5.]
        if df.empty:
            continue

        parameters_file = file[0:-21]+'_calibration_parameters.csv'
        df_para = pd.read_csv(parameters_file)
        doy = os.path.basename(file)[10:17]
        date = datetime.datetime.strptime(doy, '%Y%j')
        date = date.strftime('%Y/%m/%d')
        
        for i in range(11):
            reflectance = df_para.loc[:,'reflectance_scales'][i] * ((df.iloc[:,i+6] - df_para.loc[:,'corrected_counts_offsets'][i]) *
                                                                 df_para.loc[:,'corrected_counts_scales'][i] -
                                                                 df_para.loc[:,'reflectance_offsets'][i])
            nreflectance = reflectance/np.cos(df.loc[:,'sza']*.01*np.pi/180.)
            df.iloc[:,i+6] = nreflectance
        df.loc[:,'date'] = date
        if ('df_' not in vars()):
            df_=df
        else:
            df_=pd.concat([df_,df],ignore_index=True)
    df_.to_csv(path_or_buf=os.path.dirname(file) + os.sep + os.path.basename(file)[0:-4] + '_nreflectance_vzaLt5.csv',index=False)  # 写入csv文件
    