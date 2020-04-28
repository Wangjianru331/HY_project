# 读取HY文件星下点数据
import numpy as np
import pandas as pd
import h5py
import glob
import os
from tqdm import tqdm
def read_HY1B_nadir(file):
    try:
        f = h5py.File(file, 'r')
        
        lat = np.array(f['Navigation Data/Latitude'][:,:])
        lon = np.array(f['Navigation Data/Longitude'][:,:])
        vza = np.array(f['Navigation Data/Satellite Zenith Angle'][:,:])
        vaa = np.array(f['Navigation Data/Satellite Azimuth Angle'][:,:])
        sza = np.array(f['Navigation Data/Solar Zenith Angle'][:,:])
        saa = np.array(f['Navigation Data/Solar Azimuth Angle'][:,:])
        
        
        out = os.path.basename(file)[17:25]
        # 宽范围搜索
        location = np.where((110<=lon)&(lon<=119)&(lat>=13)&(lat<=18)&(vza<=15))
        nir=(np.array(f['Geophysical Data/DN_865'][:,:]))[location]
        b = nir>600
        nir = nir*1.0
        nir[b] = np.nan
        mask = nir/nir
        
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
        gain = np.array(f['Calibration/Calibration Coefficients Scale factor'][:,:]).flatten()
        offset = np.array(f['Calibration/Calibration Coefficients Offsets factor'][:,:]).flatten()
        calibration_para={
        'calibration_coefficients_offset_factors': offset,
        'calibration_coefficients_scale_factors': gain,
        'time_dependent_correction_constant_terms': np.array(f['Calibration/Time-dependent Correction Constant Terms'][:,:]).flatten(),
        'time_dependent_correction_linear_coeficients': np.array(f['Calibration/Time-dependent Correction Linear Coefficients'][:,:]).flatten()
        }
        df_para = pd.DataFrame(calibration_para)
        
        bands = ['DN_412', 'DN_443', 'DN_490','DN_520','DN_565','DN_670','DN_750','DN_865']
        F0=[173.3835517,191.1745851,197.7310605,183.4959944,180.4321056,149.6390249,126.7797883,95.01985516]
        sza2 = np.array(f['Navigation Data/Solar Zenith Angle'][:,:])[location]*mask
        
        for i,band in enumerate(bands):

            if offset[i]<-100 or offset[i]>100:
                offset_=0
            else:
                offset_=offset[i]
            radiance = (np.array(f['Geophysical Data/'+band][:,:]))[location]*mask*gain[i]+offset_
            nr = np.pi*radiance/F0[i]/np.cos(sza2*np.pi/180)
            nr = nr.flatten()
            band = 'b'+band[3:]
            df[band] = nr
            mean = np.nanmean(nr)  
            median = np.nanmedian(nr)  
            std = np.nanstd(nr)  
            # ptp = np.ptp(radiance[np.where(radiance < 500)])  # 最大最小值之差
            out=out+';'+str(mean)+','+str(median)+','+str(std)  #+','+str(ptp)
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
        'calibration_coefficients_offset_factors': [999],
        'calibration_coefficients_scale_factors':[999],
        'time_dependent_correction_constant_terms':[999],
        'time_dependent_correction_linear_coeficients':[999]
         }
        df_para = pd.DataFrame(calibration_para)
    return out,df,df_para


if __name__ == '__main__':
    # 输入的书为HY1B L1A级
    filedir=r'G:\hyProject'
    files = glob.glob(filedir+os.sep+'H1B***H5')
    text = filedir+os.sep+'HY1B_nReflectanceTerm_.txt.txt'
    txt = open(text, 'w')  
    pbar = tqdm(total=len(files), desc='processing:')
    for i, file in enumerate(files):
        LT,df,df_para = read_HY1B_nadir(file)
        df.to_csv(path_or_buf=os.path.dirname(file)+os.sep+os.path.basename(file)[0:-3]+'_pixel_level_info.csv', index=False) # 写入csv文件
        df_para.to_csv(path_or_buf=os.path.dirname(file)+os.sep+os.path.basename(file)[0:-3]+'_calibration_parameters.csv', index=False) 
        txt.writelines(LT + '\n')
        if LT =='error':
            print('reading file: error')
        pbar.update(1)
    txt.close()
    pbar.close()