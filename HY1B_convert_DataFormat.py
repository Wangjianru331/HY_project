# HY-1B L1A转h5格式（hy1c） 
# 只需要HY1B L1A文件作为输入
# version;1
from scipy import interpolate
from pyhdf.SD import SD, SDC
from tqdm import tqdm
import numpy as np
import shutil
import glob
import h5py
import gc
import os


def interp(y):
    # 创建一个位置矩阵：第一个数据为7，间隔为10,查看HDF文件
    navigation_points = np.arange(7, 1665, 10)
    navigation_points_new = np.arange(1, 1665, 1)
    f = interpolate.interp1d(navigation_points, y, kind='quadratic', fill_value='extrapolate')
    ynew = f(navigation_points_new)
    return ynew


def readandwrite(file):
    outfile = os.path.dirname(file)+'/H1B_OPER_OCT_L1A_20' + os.path.basename(file)[6:12] + 'T' + os.path.basename(
              file)[12:16] + '00_'+'20' + os.path.basename(file)[6:12] + 'T' + os.path.basename(file)[12:16] + '00_' +\
              os.path.basename(file)[16:21] + '_10.h5'
#     outfile = r'O:/H1B_OPER_OCT_L1A_20' + os.path.basename(file)[6:12] + 'T' + os.path.basename(
#         file)[12:16] + '00_' + '20' + os.path.basename(file)[6:12] + 'T' + os.path.basename(file)[12:16] + '00_' + \
#               os.path.basename(file)[16:21] + '_10.h5'

    if os.access(outfile, os.R_OK):
#     if os.path.exists(outfile):
        # print('skiping')
        return

    level = file[-7:-4]
    if level == 'L1A':
        geophysical_para = ['DN_412', 'DN_443', 'DN_490', 'DN_520', 'DN_565', 'DN_670', 'DN_750', 'DN_865', 'DN_11',
                            'DN_12']  # L1A
    elif level == 'L1B':
        geophysical_para = ['L_412', 'L_443', 'L_490', 'L_520', 'L_565', 'L_670', 'L_750', 'L_865', 'L_11', 'L_12']
    try:
        f = SD(file, SDC.READ | SDC.WRITE)
    except:
        errors = glob.glob(file[0:-3] + '*')
        for error in errors:
            shutil.move(error, r'G:\toSong\error2/' + os.path.basename(error))
        return

    f_new = h5py.File(outfile, 'a')

    # Calibration
    calibration = f_new.create_group('Calibration')
    data = (f.select('Time-dependent Correction Constant Terms')).get()
    calibration.create_dataset('Calibration Coefficients Offsets factor', (data.shape[0], data.shape[1]),
                               dtype='f', data=data)
    del data
    data = (f.select('Time-dependent Correction Linear Coefficients')).get()
    calibration.create_dataset('Calibration Coefficients Scale factor', (data.shape[0], data.shape[1]),
                               dtype='f', data=data)
    del data
    data = (f.select('Mirror-side Correction Factors')).get()
    calibration.create_dataset('Mirror-side Correction Scale Factors', (data.shape[0], 1),
                               dtype='f', data=data[:,0])
    calibration.create_dataset('Mirror-side Correction Offsets Factors', (data.shape[0], 1), dtype='f', data=data[:, 1])
    del data
    data = (f.select('Time-dependent Correction Constant Terms')).get()
    calibration.create_dataset('Time-dependent Correction Constant Terms', (data.shape[0], data.shape[1]), dtype='f',
                               data=data)
    del data
    data = (f.select('Time-dependent Correction Linear Coefficients')).get()
    calibration.create_dataset('Time-dependent Correction Linear Coefficients', (data.shape[0], data.shape[1]),
                               dtype='f', data=data)
    del data
    data = (f.select('Time-dependent Correction Quadratic Coefficients')).get()
    calibration.create_dataset('Time-dependent Correction Quadratic Coefficients', (data.shape[0], data.shape[1]),
                               dtype='f', data=data)
    del data
    data = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    calibration.create_dataset('Vicarious Calibration gan factor', (data.shape[0], data.shape[1]), dtype='f', data=data)
    del data
    calibration.attrs['Calibration Entry Year'] = 2019
    calibration.attrs['Calibration Entry Day'] = 330
    calibration.attrs['Calibration Reference Year'] = 0
    calibration.attrs['Calibration Reference Day'] = 0
    calibration.attrs['Calibration Reference Minute'] = 0
    calibration.attrs['Visible Channel Radiance Data Unit'] = np.string_('mWcm-2 um-1 sr-1')
    calibration.attrs['Infrared Channel Radiance Data Unit'] = np.string_('mWcm-2 um-1 sr-1')

    # Extra Data
    group = f_new.create_group('Extra Data')
    bb = (f.select('Blackbody Pixels')).get()
    ds = (f.select('Space Pixels')).get()
    Ext_xxx = ['412', '443', '490', '520', '565', '670', '750', '865', '11', '12']
    for j, ext_x in enumerate(Ext_xxx):
        data = np.zeros(shape=(ds.shape[0], 43))
        data[:, 10: 20] = ds[:, j*10:(j+1)*10]
        data[:, 0: 10] = bb[:, j * 10:(j + 1) * 10]
        ext = group.create_dataset('Ext_'+ext_x, (data.shape[0], 43), dtype='uint16', data=data)
        if len(ext_x) == 2:
            ext.attrs['long_name'] = np.string_('B'+ext_x+'um Extra data counts')
        else:
            ext.attrs['long_name'] = np.string_('B'+ext_x+'nm Extra data counts')
    del group,ext

    # geophysical Data
    group = f_new.create_group('Geophysical Data')
    for ID in geophysical_para:
        data = (f.select(ID)).get()
        dataset=group.create_dataset(ID, (data.shape[0], data.shape[1]), dtype='uint16', data=data)
        dataset.attrs['Unit'] = np.string_('None')
        dataset.attrs['long_name'] = np.string_('Top of Atmosphere B'+ID[3:]+'nm radiance counts')
    del group, dataset

    # navigaton data
    parameters = ['Latitude', 'Longitude', 'Solar Zenith Angle', 'Solar Azimuth Angle', 'Satellite Zenith Angle',
                  'Satellite Azimuth Angle']
    group = f_new.create_group('Navigation Data')
    for ID in parameters:
        parameter = f.select(ID)[:, :]
        nl = np.arange(parameter.shape[0])
        # 当经度跨过180度经线时，
        if ID == 'Longitude':
            if parameter.max() - parameter.min() > 300:  # 跨180经度时
                parameter[parameter < 0] = parameter[parameter < 0] + 360.0  # 180经线两边数值连续，以正确插值
                parameter_inter = np.array([*map(lambda n: interp(parameter[n, :]), nl)])
                parameter_inter[parameter_inter > 180] = parameter_inter[parameter_inter > 180] - 360.0  # 变回来
                group.create_dataset(ID, (data.shape[0], data.shape[1]), dtype='f', data=parameter_inter)
                continue
        # 按行计算
        parameter_inter = np.array([*map(lambda n: interp(parameter[n, :]), nl)])
        group.create_dataset(ID, (data.shape[0], data.shape[1]), dtype='f', data=parameter_inter)

    group.attrs['Navigation Point Counts'] = np.string_(str(parameter_inter.shape[1]))
    group.attrs['First Navigation Points'] = np.string_('1')
    group.attrs['Pixel-intervals of Navigation Point'] = np.string_('1')
    del group, parameters, data

    # QC Attributes
    group = f_new.create_group('QC Attributes')
    parameters = ['Staturated Pixel Counts', 'Zero Pixel Counts']
    for ID in parameters:
        data = (f.select(ID)).get()
        group.create_dataset(ID, (data.shape[0], data.shape[1]), dtype='uint16', data=data)
    group.attrs['Missing Frame Counts'] = np.int32(0)
    del group, data

    # Scan Line Attributes
    group = f_new.create_group('Scan Line Attributes')
    data = (f.select('Attitude Parameters')).get()
    group.create_dataset('Attitude Parameters', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Center Latitude')).get()
    group.create_dataset('Center Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Center Longitude')).get()
    group.create_dataset('Center Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Center Solar Zenith')).get()
    group.create_dataset('Center Solar Zenith Angle', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Start Latitude')).get()
    group.create_dataset('Start Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Start Longitude')).get()
    group.create_dataset('Start Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('End Latitude')).get()
    group.create_dataset('End Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('End Longitude')).get()
    group.create_dataset('End Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    del data

    data = (f.select('Frame Number')).get()
    frame = (np.arange(data.shape[0]))%8
    frame.shape=(frame.shape[0], 1)
    group.create_dataset('Frame Number', (frame.shape[0], frame.shape[1]), dtype='int16', data=frame)
    del data

    data = (f.select('Infrared Channel Calibration Data')).get()
    group.create_dataset('Infrared Channel Calibration Data', (data.shape[0], data.shape[1], data.shape[2]),
                         dtype=type(data[0][0][0]), data=data)
    del data

    data = (f.select('Millisecond')).get()
    millisecond = group.create_dataset('Millisecond', (data.shape[0], data.shape[1]), dtype=np.float64, data=data)
    millisecond.attrs['Fillin_value'] = -999.0
    millisecond.attrs['Unit'] = np.string_('milliseconds since at 00:00:00 on this day ')
    del data

    data = (f.select('Mirror-side Flag')).get()
    group.create_dataset('Mirror-side Flag', (data.shape[0], data.shape[1]), dtype='|S1', data=data)
    del data
    # no ORB_VEC dataset in HDF
    # data = (f.select('ORB_VEC')).get()
    # group.create_dataset('ORB_VEC', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
    # del data
    group.attrs['Instrument Parameters'] = np.string_('412nm,443nm,490nm,520nm,565nm,670nm,750nm,865nm,10.3-11.4um,'
                                                      '11.5-12.5um')

    # auxiliary
    attrs = f.attributes()
    for key, value in attrs.items():
        if type(value)==str:
            f_new.attrs[key] = np.string_(value)
        else:
            f_new.attrs[key] = value
    f_new.attrs['DayorNight'] = np.string_('D')

    f.end()
    f_new.close()
    return outfile


def l1a2l1b(infile):
    # infile = r'G:\H5TEST\HY-1B/H1B_OPER_OCT_L1A_20070513T022200_20070513T022200_00457_10.h5'
    f = h5py.File(infile, 'r')
    outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:13] + 'L1B' + os.path.basename(infile)[16:]
    f_new = h5py.File(outfile, 'w')

    # 文件属性
    for attr_of_file in f.attrs.items():
        f_new.attrs.create(attr_of_file[0], attr_of_file[1], shape=attr_of_file[1].shape, dtype=attr_of_file[1].dtype)

    # 组
    for group_of_file in f.items():
        group = f_new.create_group(group_of_file[0])

        # 组属性
        for attr_of_group in group_of_file[1].attrs.items():
            group.attrs.create(attr_of_group[0], attr_of_group[1], shape=attr_of_group[1].shape,
                               dtype=attr_of_group[1].dtype)
            del attr_of_group

        # 组数据集
        for i, dataset_of_group in enumerate(group_of_file[1].items()):

            if dataset_of_group[0][:2] == 'DN':
                if dataset_of_group[0] != 'DN_11' and dataset_of_group[0] != 'DN_12':
                    dataset = group.create_dataset('L' + dataset_of_group[0][2:], dataset_of_group[1].shape,
                                                   dtype=np.float32,
                                                   data=dataset_of_group[1][()] *
                                                        f['Calibration/Calibration Coefficients Scale factor'][i - 2])
            else:
                dataset = group.create_dataset(dataset_of_group[0], dataset_of_group[1].shape,
                                               dtype=dataset_of_group[1].dtype, data=dataset_of_group[1][()])
            # 数据集属性
            for attr_of_dataset in dataset_of_group[1].attrs.items():
                dataset.attrs.create(attr_of_dataset[0], attr_of_dataset[1], shape=attr_of_dataset[1].shape,
                                     dtype=attr_of_dataset[1].dtype)

    return


if __name__ == '__main__':
    files = glob.glob(r'I:\HY_project\HY1BL1A/H1B*.L1A.HDF')

    pbar = tqdm(total=len(files), desc='processing:')
    for i, file in enumerate(files):
        pbar.update(1)
        outfile = readandwrite(file)
        # l1a2l1b(outfile)

        # gc.collect()
    pbar.close()