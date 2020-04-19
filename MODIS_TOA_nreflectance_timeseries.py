from pyhdf.SD import SD, SDC
from tqdm import tqdm
import pandas as pd
import glob
import os
import numpy as np
import datetime
def read_nadir_modis(file):
    try:
        f = SD(file, SDC.READ | SDC.WRITE)
        lat = (f.select('Latitude')).get()
        lon = (f.select('Longitude')).get()
        vza = (f.select('SensorZenith')).get()
        vaa = (f.select('SensorAzimuth')).get()
        sza = (f.select('SolarZenith')).get()
        saa = (f.select('SolarAzimuth')).get()
        Lt = (f.select('EV_1KM_RefSB'))

        out = datetime.datetime.strptime(os.path.basename(file)[10:17], '%Y%j').date().strftime('%Y%m%d')
        location = np.where((110<=lon)&(lon<=111)&(lat>=15)&(lat<=16.5)&(vza<=10))
        DN = (Lt.get()[10,:,:][location]-Lt.attributes()['corrected_counts_offsets'][10])*Lt.attributes()['corrected_counts_scales'][10]
        nir = Lt.attributes()['radiance_scales'][10]*(DN-Lt.attributes()['radiance_offsets'][10])
        b = nir>8
        nir[b] = np.nan
        mask = nir/nir              # 设置耀斑云等无效值阈值
        
        lon = lon[location]*mask
        lat = lat[location]*mask
        vza = vza[location]*mask
        vaa = vaa[location]*mask
        sza = sza[location]*mask
        saa = saa[location]*mask
        
        lon = lon.flatten()
        lat = lat.flatten()
        vza = vza.flatten()
        vaa = vaa.flatten()
        sza = sza.flatten()
        saa = saa.flatten()
        
        data = {'lat':lat,
                'lon':lon,
                'vza':vza,
                'vaa':vaa,
                'sza':sza,
                'saa':saa
                }
        df = pd.DataFrame(data)
        
        calibration_para={
        'corrected_counts_scales': Lt.attributes()['corrected_counts_scales'],
        'corrected_counts_offsets':Lt.attributes()['corrected_counts_offsets'],
        'reflectance_scales':Lt.attributes()['reflectance_scales'],
        'reflectance_offsets':Lt.attributes()['reflectance_scales'],
        'radiance_scales':Lt.attributes()['radiance_scales'],
        'radiance_offsets':Lt.attributes()['radiance_offsets']}
        df_para = pd.DataFrame(calibration_para)
        sza2 = (f.select('SolarZenith')).get()
        sza2 = sza2[location]*mask
        cos = np.cos(sza2*np.pi/180)
        for i in range(11):
            a = Lt.get()[i,:,:][location]*mask
            a = np.array(a, dtype=float)
            fill = Lt.attributes()['_FillValue']
            b = a==fill
            a[b]=np.nan
            a = a.flatten()
            band = 'b'+str(i)
            df[band] = a
            
            result = Lt.attributes()['reflectance_scales'][i]*((a-Lt.attributes()['corrected_counts_offsets'][i])*Lt.attributes()['corrected_counts_scales'][i]-Lt.attributes()['reflectance_offsets'][i])
            result = result/cos
            mean = round(np.nanmean(result),4)            
            median = round(np.nanmedian(result),4)  
            std = round(np.nanstd(result),4)             
            out=out+';'+str(mean)+','+str(median)+','+str(std)  #+','+str(ptp)
        df.dropna(axis=0,how='any')    
        return out,df,df_para
    except:
        out ='error'
        data = {'lat':[999],
                'lon':[999],
                'vza':[999],
                'vaa':[999],
                'sza':[999],
                'saa':[999]
                }
        df = pd.DataFrame(data)
         calibration_para={
        'corrected_counts_scales': [999],
        'corrected_counts_offsets':[999],
        'reflectance_scales':[999],
        'reflectance_offsets':[999],
        'radiance_scales':[999],
        'radiance_offsets':[999]}
        df_para = pd.DataFrame(calibration_para)
        return out,df,df_para
		
		
if __name__ == '__main__':
    # 该程序返回三个文件，每个文件对应的定标参数，每个文件对应的满足要求的像元信息，
    #所有文件的归一化天顶反射率时间序列结果，各波段的包括均值中值标准差
    filedir=r'H:\MODIS'
    files = glob.glob(filedir+os.sep+'MOD021KM*hdf')
    txt = open(filedir+os.sep+'Modis_nReflectanceTerm_.txt', 'w')                            
    pbar = tqdm(total=len(files), desc='processing:')
    for i, file in enumerate(files):
        LT,df,df_para = read_nadir_modis(file)
        df.to_csv(path_or_buf=os.path.dirname(file)+os.sep+os.path.basename(file)[0:-4]+'_pixel_level_info.csv', index=False) # 写入csv文件
        df_para.to_csv(path_or_buf=os.path.dirname(file)+os.sep+os.path.basename(file)[0:-4]+'_calibration_parameters.csv', index=False) # 写入csv文件
#         print(os.path.basename(file))
        txt.writelines(LT + '\n')
        pbar.update(1)
    txt.close()
    pbar.close()