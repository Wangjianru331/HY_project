import xml.etree.cElementTree as ET
import pyorbital.orbital as orbital
from scipy import interpolate
from pysolar.solar import *
# import PIL.Image as Image
from lxml import etree
from tqdm import tqdm
# import scipy.ndimage
import pandas as pd
import numpy as np
# import pyorbital
import datetime
import shutil
import struct
# import scipy
import h5py
import glob
import bz2
import os
import cv2
# import gc


class L1bparase:

    def read_head(self, HY1AL1Bfile_illustrate_file):
        # 使用xml存储文件信息
        # 生成根节点

        df = pd.read_excel(HY1AL1Bfile_illustrate_file, sheet_name="head",
                           index_col=None)  # 关于扫描行的数据说明
        root = etree.Element('root', attrib={'version': '1.2', 'tag': 'HY1AL1B'})
        # 生成第一个子节点 head
        head = etree.SubElement(root, 'head')

        # head 节点的子节点
        for i in range(df.shape[0]):
            start1 = int(df.loc[i, "startingPosition"]) - 1  # 起始位置
            bytesNum1 = int(df.loc[i, "bytes_number"])
            stop1 = start1 + bytesNum1  # 结束位置
            para_type1 = str(df.loc[i, "type"])
            type_length1 = int(df.loc[i, "type_length"])
            byte_order1 = df.loc[i, "byte_order"]

            if para_type1 == "s":
                para = self.parse_character(start1, stop1, para_type1, type_length1, bytesNum1, byte_order1)
            else:
                para = struct.unpack(byte_order1 + para_type1 * int((bytesNum1 / type_length1)), lines[start1:stop1])

            if isinstance(para, tuple) or isinstance(para, list):  # 将由单个字符组成的元组组成一个字符串
                para_result = ""
                for index1, zz in enumerate(para):
                    para_result = para_result + ',' + str(zz)
                para = para_result

            etree.SubElement(head, (df.loc[i, "parameter_name"]).replace(' ', '')).text = para.strip(
                ',')  # .decode("UTF-8")
            tree = etree.ElementTree(root)
            tree.write(outxml, pretty_print=True, xml_declaration=True, encoding='utf-8')


    def func(self, row):
        # 图像数据
        value = struct.unpack(byte_order + para_type * int((bytesNum / type_length)),
                              lines[5120 + row * 43008 + start:5120 + row * 43008 + stop])
        value_0 = np.array(value)
        # 划分为1024行，10列的数组，每行代表10个波段，因此，从中取出第一列再转置就生成了第一个波段的第一行数据
        value_1 = value_0.reshape((width, depth))
        # 切记，转置是以左上角为原点，逆时针转90°
        return value_1.T

    def parse_character(self, start, stop, para_type, type_length, bytesNum, byte_order):
        para = struct.unpack(byte_order + para_type * int((bytesNum / type_length)), lines[start:stop])
        para_result = ''
        for zz in para:
            if 31 < ord(zz) < 127:
                para_result = para_result + str(zz.decode('utf8'))
        para = para_result
        if para.strip() == "":
            para = u"NoneValue"
        if isinstance(para, tuple) or isinstance(para, list):  # 将由单个字符组成的元组组成一个字符串
            para_result = ""
            for index, zz in enumerate(para):
                if index == 0:
                    para_result = str(zz.decode('utf8'))
                else:
                    para_result = para_result + ',' + str(zz.decode('utf8'))
            para = para_result
        return para

    def read_write(self, file, HY1AL1Bfile_illustrate_file):
        #         print(file)

        #                 file = "F:\HY1A1B\H1ACBD0215200242.L1B.bz2"
       # try:
            binfile = bz2.BZ2File(file, "rb")
            global lines, outxml
            lines = binfile.read()

            filepath, filename = os.path.split(file)
            outfile = filepath + '/' + filename[0:-4] + '.H5'
            outfile1 = filepath + '/' + filename[0:-8] + '.jpg'
            outxml = filepath + '/' + filename[0:-4] + '.xml'

            #           头文件的信息解析到xml中去
            if os.access(outfile, os.R_OK):
                return outfile
            else:
                self.read_head(HY1AL1Bfile_illustrate_file)
                h5file = h5py.File(outfile, 'w')
                data_illustrate = pd.read_excel(HY1AL1Bfile_illustrate_file, sheet_name="data",
                                                index_col=None)  # 关于扫描行的数据说明
                global columns
                columns = struct.unpack(">" + 'l', lines[1308:1312])[0]  # 每行的像元数,即列数
                rows = struct.unpack(">" + 'l', lines[1312:1316])[0]  # 本景的扫描行数
                image_terms = ["L_412", "L_443", "L_490", "L_520", "L_565", "L_670", "L_750", "L_865", "L_11", "L_12"]
                latlon_terms = ["latitude", "longitude"]
                sensor_matrix = ["1_parameters", "2_parameters", "3_parameters"]

                for i in range(data_illustrate.shape[0]):
                    #               设置全局参数，传参简单一些

                    global start, stop, depth, width, para_type, type_length, bytesNum, byte_order
                    start = int(data_illustrate.loc[i, "startingPosition"]) - 1  # 起始位置
                    bytesNum = int(data_illustrate.loc[i, "bytes_number"])
                    stop = start + bytesNum  # 结束位置
                    para_type = str(data_illustrate.loc[i, "type"])
                    type_length = int(data_illustrate.loc[i, "type_length"])
                    byte_order = data_illustrate.loc[i, "byte_order"]

                    #               获取解析目标的排列信息
                    depth = int(data_illustrate.loc[i, "depth"])
                    width = int(data_illustrate.loc[i, "width"])
                    parameter_name = data_illustrate.loc[i, "parameter_name"]

                    #               if parameter_name == "Attitude parameter":
                    #                   print("Attitude parameter")
                    #                   print(result_list)

                    #              如果数据是字符型，则忽略
                    if para_type == "s":
                        continue
                    #                     result_list = parse_character(start, stop, para_type, type_length, bytesNum, byte_order)
                    #                     print(result_list)
                    else:
                        row = list(range(rows))
                        #                  按照相应的规律读取数据，并按照相应的规律进行行列波段的调整
                        result = map(lambda x: self.func(x), row)
                        result_list = list(result)

                    parameter = h5file.create_group(parameter_name)  # 开始准备写入hdf5
                    terms = []
                    terms.append(parameter_name)
                    if parameter_name == "Scan image data":
                        terms = image_terms
                    if parameter_name == "Pixels location data":
                        terms = latlon_terms
                    if parameter_name == "Sensor matrix":
                        terms = sensor_matrix
                    for j in range(depth):
                        if depth > 1:
                            t = np.array(result_list)[:, j]
                        else:
                            t = np.array(result_list)
                        parameter.create_dataset(terms[j], (rows, width), data=t)

                #           选出快视图的波段
                L750 = h5file['Scan image data/L_750'][()]
                arr = np.zeros((L750.shape[0], L750.shape[1], 3))
                arr[:, :, 0] = L750
                arr[:, :, 1] = h5file['Scan image data/L_565'][()]
                arr[:, :, 2] = h5file['Scan image data/L_490'][()]
                res = cv2.resize(arr, dsize=(int(L750.shape[1] / 4), int(L750.shape[0] / 4)),
                                 interpolation=cv2.INTER_NEAREST)

                outfile1 = outfile1.replace('\\', '/')

                # scipy.misc.imsave(outfile1,res)
                cv2.imwrite(outfile1,res)

                h5file.close()
                return outfile
       # except:
        #    return 'error'


class L0parase:
    #     这里仅用到L0的冷空黑体数据
    def image_func(self, bs_list):
        #         图像数据，在这个程序里面没用到，直接用了L1B的图像数据
        #         输入每一帧的bit级数据，输出为重新排列的数据

        #         将数据分为每10个二进制串为组，即每个数据是10bit的
        bs_list = [bs_list[i:i + 10] for i in range(0, len(bs_list), 10)]
        #         二进制转为10进制数
        value = np.array([int(x, 2) for x in bs_list])
        #         排列
        return np.reshape(value, (10, 4, 1024), order='F')

    def space_black_func(self, bs_list):
        #         输入每一帧的bit级数据，输出为重新排列的数据

        #         将数据分为每10个二进制串为组，即每个数据是10bit的
        bs_list = [bs_list[i:i + 10] for i in range(0, len(bs_list), 10)]
        #         二进制转为10进制数
        value = np.array([int(x, 2) for x in bs_list])
        #         排列
        return np.reshape(value, (10, 4, 6), order='F')

    def run_this_function(self, infile):
        binFile = bz2.BZ2File(infile, "rb")
        binFile.seek(0)
        context = binFile.read()
        #         读入的为16进制字符串，将其全部转为二进制字符串，前两个为二进制标识，将其去掉
        all_binary_string = bin(int.from_bytes(context, byteorder='big'))[2:]

        #         共有frame_number帧数据
        frame_number = int(len(all_binary_string) / 419904)

        #         parameter = h5file.create_group('Geophysical Data')  # 开始准备写入hdf5
        bands = ["412", "443", "490", "520", "565", "670", "750", "865", "11", "12"]

        #         黑体信号
        bs_list = [*map(lambda x: all_binary_string[419904 * x + 60 + 100:419904 * x + 60 + 2500], range(frame_number))]
        mat = [*map(lambda x: self.space_black_func(x), bs_list)]
        black_new = np.concatenate((mat), axis=1)

        #         冷空信号
        bs_list = [
            *map(lambda x: all_binary_string[419904 * x + 60 + 2500:419904 * x + 60 + 4900], range(frame_number))]
        mat = [*map(lambda x: self.space_black_func(x), bs_list)]
        space_new = np.concatenate((mat), axis=1)
        data = np.zeros(shape=(10, space_new.shape[1], 43))
        data[:, :, 0: 6] = space_new
        data[:, :, 10: 16] = black_new
        return data


class to1cformat():
    #     把数据转为1c星格式
    def interp(self, y):
        #         创建一个位置矩阵：第一个数据为7，间隔为10,查看HDF文件
        navigation_points = np.arange(4, 1024, 10)
        navigation_points_new = np.arange(0, 1024, 1)
        f = interpolate.interp1d(navigation_points, y, kind='quadratic', fill_value='extrapolate')
        ynew = f(navigation_points_new)
        return ynew

    def read2h5(self, infile):
        # 主程序
        #         print(infile)

        try:
            f = h5py.File(infile, 'r')
        except:
            errordir = os.path.dirname(infile) + os.sep + 'error'
            if not os.path.exists(errordir):
                os.makedirs(errordir)
            errors = glob.glob(infile[0:-3] + '*')
            for error in errors:
                shutil.move(error, errordir + os.sep + os.path.basename(error))
            return

        #         输出文件名
        millisecond = f['Millisecond/Millisecond'][()]
        hour = np.min(millisecond) // (1000 * 60 * 60)
        minute = (np.min(millisecond) - hour * (1000 * 60 * 60)) // (1000 * 60)
        second = (np.min(millisecond) - hour * (1000 * 60 * 60) - minute * 1000 * 60) // (1000)

        time1 = datetime.datetime(int('20' + os.path.basename(infile)[6:8]), 1, 1) + \
                datetime.timedelta(days=int(os.path.basename(infile)[8:11]) - 1) + \
                datetime.timedelta(hours=int(hour)) + datetime.timedelta(minutes=int(minute)) + \
                datetime.timedelta(seconds=int(second))

        hour = np.max(millisecond) // (1000 * 60 * 60)
        minute = (np.max(millisecond) - hour * (1000 * 60 * 60)) // (1000 * 60)
        second = (np.max(millisecond) - hour * (1000 * 60 * 60) - minute * 1000 * 60) // (1000)

        time2 = datetime.datetime(int('20' + os.path.basename(infile)[6:8]), 1, 1) + \
                datetime.timedelta(days=int(os.path.basename(infile)[8:11]) - 1) + \
                datetime.timedelta(hours=int(hour)) + datetime.timedelta(minutes=int(minute)) + \
                datetime.timedelta(seconds=int(second))

        # del millisecond

        tree = ET.ElementTree(file=infile[0:-2] + 'xml')
        root = tree.getroot()
        node = root[0].find("OrbitNumber")

        outfile = os.path.dirname(infile) + os.sep + 'H1A_OPER_OCT_L1B_' + time1.strftime("%Y%m%dT%H%M%S") + '_' + \
                  time2.strftime("%Y%m%dT%H%M%S") + '_' + node.text + '_10.h5'

        if os.access(outfile, os.R_OK):
            return outfile

        f_new = h5py.File(outfile, 'a')

        # Calibration
        calibration = f_new.create_group('Calibration')
        data = np.zeros((10, 1))
        calibration.create_dataset('Calibration Coefficients Offsets factor', (data.shape[0], data.shape[1]),
                                   dtype='f', data=data)
        del data
        data = np.ones((10, 1))
        calibration.create_dataset('Calibration Coefficients Scale factor', (data.shape[0], data.shape[1]),
                                   dtype='f', data=data)
        del data
        data = np.ones((10, 1))
        calibration.create_dataset('Mirror-side Correction Scale Factors', (data.shape[0], 1),
                                   dtype='f', data=data[:, 0])
        data = np.zeros((10, 1))
        calibration.create_dataset('Mirror-side Correction Offsets Factors', (data.shape[0], 1), dtype='f',
                                   data=data[:, 0])
        del data
        data = np.zeros((10, 1))
        calibration.create_dataset('Time-dependent Correction Constant Terms', (data.shape[0], data.shape[1]),
                                   dtype='f',
                                   data=data)
        del data
        data = np.ones((10, 1))
        calibration.create_dataset('Time-dependent Correction Linear Coefficients', (data.shape[0], data.shape[1]),
                                   dtype='f', data=data)
        del data
        data = np.ones((10, 1))
        calibration.create_dataset('Time-dependent Correction Quadratic Coefficients', (data.shape[0], data.shape[1]),
                                   dtype='f', data=data)
        del data
        data = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
        calibration.create_dataset('Vicarious Calibration gan factor', (data.shape[0], data.shape[1]), dtype='f',
                                   data=data)
        del data
        calibration.attrs['Calibration Entry Year'] = np.int16(2019)
        calibration.attrs['Calibration Entry Day'] = np.int16(330)
        calibration.attrs['Calibration Reference Year'] = np.int16(0)
        calibration.attrs['Calibration Reference Day'] = np.int16(0)
        calibration.attrs['Calibration Reference Minute'] = np.int32(0)
        calibration.attrs['Visible Channel Radiance Data Unit'] = np.string_('mWcm-2 um-1 sr-1')
        calibration.attrs['Infrared Channel Radiance Data Unit'] = np.string_('mWcm-2 um-1 sr-1')

        # geophysical Data
        group = f_new.create_group('Geophysical Data')
        for j, dataset_of_group in enumerate(f['Scan image data'].items()):
            dataset = group.create_dataset(dataset_of_group[0], dataset_of_group[1].shape,
                                           dtype=np.float32, data=dataset_of_group[1][()])
            dataset.attrs['Unit'] = np.string_('None')
            dataset.attrs['long_name'] = np.string_(
                'Top of Atmosphere B' + dataset_of_group[0][2:] + 'nm/um radiance counts')
        del group, dataset

        # Extra Data
        group = f_new.create_group('Extra Data')
        Ext_xxx = ['412', '443', '490', '520', '565', '670', '750', '865', '11', '12']
        try:
            L0file = glob.glob(os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-6] + 'L0.bz2')[0]
            L0 = L0parase()
            data = L0.run_this_function(L0file)
        #         print('LA')
        except:
            data = np.zeros(shape=(10, dataset_of_group[1].shape[0], 43))
        for j, ext_x in enumerate(Ext_xxx):
            ext = group.create_dataset('Ext_' + ext_x, (data.shape[1], 43), dtype='uint16', data=data[j, :, :])
            if len(ext_x) == 2:
                ext.attrs['long_name'] = np.string_('B' + ext_x + 'um Extra data counts')
            else:
                ext.attrs['long_name'] = np.string_('B' + ext_x + 'nm Extra data counts')
        del group, ext

        # navigaton data
        group = f_new.create_group('Navigation Data')

        # 计算四个角度

        # =======================================================================================================================
        lat = f['Pixels location data/latitude'][()]
        lon = f['Pixels location data/longitude'][()]
        millisecond = f['Millisecond/Millisecond'][()]
        # year = np.ones(shape=lat.shape) * int('20' + os.path.basename(infile)[6:8])
        # # year= np.array([[2003,2003],[2003,2004]])
        # # month=np.array([[1,2],[6,12]])
        # # day=np.array([[2,16],[14,2]])
        # DOY = np.ones(shape=lat.shape) * int(os.path.basename(infile)[8:11])
        #
        # millisecond = np.repeat(millisecond, 102, axis=1)
        # # hour = np.array([[1, 2], [3, 4]])
        # hour = np.trunc(millisecond / 1000 / 3600)
        # minu = np.trunc((millisecond - hour * 1000 * 3600) / 1000 / 60)
        # sec = (millisecond - hour * 1000 * 3600 - minu * 1000 * 60) / 1000
        # # minu=np.array([[23,23],[23,22]])
        # # sec=np.array([[1,2],[24,55]])
        #
        # TimeZone = np.trunc((lon - np.sign(lon) * 7.5) / 15 + np.sign(lon))
        #
        # # N0   sitar=θ
        # N0 = 79.6764 + 0.2422 * (year - 1985) - np.trunc((year - 1985) / 4.0)
        # sitar = 2 * np.pi * (DOY - N0) / 365.2422
        # ED1 = 0.3723 + 23.2567 * np.sin(sitar) + 0.1149 * np.sin(2 * sitar) - 0.1712 * np.sin(
        #     3 * sitar) - 0.758 * np.cos(
        #     sitar) + 0.3656 * np.cos(2 * sitar) + 0.0201 * np.cos(3 * sitar)
        # ED = ED1 * np.pi / 180  # ED本身有符号
        #
        # dLon = (lon - TimeZone * 15.0) * np.sign(lon)
        #
        # # 时差
        # Et = 0.0028 - 1.9857 * np.sin(sitar) + 9.9059 * np.sin(2 * sitar) - 7.0924 * np.cos(sitar) - 0.6882 * np.cos(
        #     2 * sitar)
        # gtdt1 = hour + minu / 60.0 + sec / 3600.0 + dLon / 15  # 地方时
        # gtdt = gtdt1 + Et / 60.0
        # dTimeAngle1 = 15.0 * (gtdt - 12)
        # dTimeAngle = dTimeAngle1 * np.pi / 180
        # latitudeArc = lat * np.pi / 180
        #
        # # 高度角计算公式
        # HeightAngleArc = np.arcsin(
        #     np.sin(latitudeArc) * np.sin(ED) + np.cos(latitudeArc) * np.cos(ED) * np.cos(dTimeAngle))
        # # 方位角计算公式
        # CosAzimuthAngle = (np.sin(HeightAngleArc) * np.sin(latitudeArc) - np.sin(ED)) / np.cos(HeightAngleArc) / np.cos(
        #     latitudeArc)
        # AzimuthAngleArc = np.arccos(CosAzimuthAngle)
        # HeightAngle = HeightAngleArc * 180 / np.pi
        # sza = 90 - HeightAngle
        # AzimuthAngle1 = AzimuthAngleArc * 180 / np.pi
        # saa = 180 + AzimuthAngle1 * np.sign(dTimeAngle)
        # ========================================================================================================================
        # 使用pysolar计算sza saa
        time = [*map(
            lambda t: datetime.datetime(int('20' + os.path.basename(infile)[6:8]), 1, 1, tzinfo=datetime.timezone.utc) + \
                      datetime.timedelta(days=(int(os.path.basename(infile)[8:11]) - 1)) + \
                      datetime.timedelta(milliseconds=t[0]), millisecond.tolist())]

        time = np.repeat(np.array(time).reshape(-1, 1), 102, axis=1)
        time = time.flatten()
        lat = lat.flatten()
        lon = lon.flatten()

        sza = np.array([*map(lambda sx, sy, t: 90 - get_altitude(sx, sy, t), lat, lon, time)])
        sza = sza.reshape(-1, 102)
        saa = np.array([*map(lambda sx, sy, t: get_azimuth(sx, sy, t), lat, lon, time)])
        saa = saa.reshape(-1, 102)

        # sza = np.array([*map(lambda x, y, t: 90.0 - get_altitude(x, y, t), lat, lon, time)]).reshape(-1, 102)
        # saa = np.array([*map(lambda x, y, t: get_azimuth(x, y, t), lat, lon, time)]).reshape(-1, 102)
        center_lat = f['Center Latitude/Center Latitude'][()]
        center_lon = f['Center Longitude/Center Longitude'][()]
        center_lat = np.repeat(center_lat, 102, axis=1)
        center_lon = np.repeat(center_lon, 102, axis=1)
        center_lat = center_lat.flatten()
        center_lon = center_lon.flatten()
        # pyorbital.orbital.get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt)
        view_angle = np.array(
            [*map(lambda sx, sy, t, x, y: orbital.get_observer_look(np.atleast_1d(sx), np.atleast_1d(sy),
                                                                    np.atleast_1d(798), t,
                                                                    np.atleast_1d(x),
                                                                    np.atleast_1d(y), np.atleast_1d(0)),
                  center_lon, center_lat, time, lon, lat)])
        vaa = (view_angle[:, 0]).reshape(-1, 102)
        vza = (90 - view_angle[:, 1]).reshape(-1, 102)
        lon = lon.reshape(-1, 102)
        lat = lat.reshape(-1, 102)

        parameters = [lon, lat, sza, saa, vza, vaa]
        parameter_ID = ['Longitude', 'Latitude', 'Solar Zenith Angle', 'Solar Azimuth Angle', 'Satellite Zenith Angle',
                        'Satellite Azimuth Angle']

        for ID, parameter in enumerate(parameters):
            nl = np.arange(parameter.shape[0])
            #             当经度跨过180度经线时，
            if parameter_ID[ID] == 'Longitude':
                if parameter.max() - parameter.min() > 300:  # 跨180经度时
                    parameter[parameter < 0] = parameter[parameter < 0] + 360.0  # 180经线两边数值连续，以正确插值
                    parameter_inter = np.array([*map(lambda n: self.interp(parameter[n, :]), nl)])
                    parameter_inter[parameter_inter > 180] = parameter_inter[parameter_inter > 180] - 360.0  # 变回来
                    group.create_dataset(parameter_ID[ID], (parameter_inter.shape[0], parameter_inter.shape[1]),
                                         dtype=np.float32,
                                         data=parameter_inter)
                    continue  # 经度无需再写
            #             按行计算
            parameter_inter = np.array([*map(lambda n: self.interp(parameter[n, :]), nl)])
            group.create_dataset(parameter_ID[ID], (parameter_inter.shape[0], parameter_inter.shape[1]), dtype='f',
                                 data=parameter_inter)

        group.attrs['Navigation Point Counts'] = np.int32(parameter_inter.shape[1])
        group.attrs['First Navigation Points'] = np.int32(1)
        group.attrs['Pixel-intervals of Navigation Point'] = np.int32(1)
        del group, parameters

        #         QC Attributes
        group = f_new.create_group('QC Attributes')
        parameters = ['Staturated Pixel Counts', 'Zero Pixel Counts']
        data = f['Saturated Pixels/Saturated Pixels'][()]
        group.create_dataset('Staturated Pixel Counts', (data.shape[0], data.shape[1]), dtype='uint16', data=data)
        data = f['Zero Pixels/Zero Pixels'][()]
        group.create_dataset('Zero Pixel Counts', (data.shape[0], data.shape[1]), dtype='uint16', data=data)
        group.attrs['Missing Frame Counts'] = np.int32(0)
        del group, data

        #         Scan Line Attributes
        group = f_new.create_group('Scan Line Attributes')
        # data = (f.select('Attitude Parameters')).get()
        # group.create_dataset('Attitude Parameters', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        # del data
        data = f['Center Latitude/Center Latitude'][()]
        group.create_dataset('Center Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        del data

        data = f['Center Latitude/Center Latitude'][()]
        group.create_dataset('Center Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        del data

        data = f['Center Solar Zenith/Center Solar Zenith'][()]
        group.create_dataset('Center Solar Zenith Angle', (data.shape[0], data.shape[1]), dtype=type(data[0][0]),
                             data=data)
        del data

        data = f['Start Latitude/Start Latitude'][()]
        group.create_dataset('Start Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        del data

        data = f['Start Longitude/Start Longitude'][()]
        group.create_dataset('Start Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        del data

        data = f['End Latitude/End Latitude'][()]
        group.create_dataset('End Latitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        del data

        data = f['End Longitude/End Longitude'][()]
        group.create_dataset('End Longitude', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)

        frame = (np.arange(data.shape[0])) % 8
        frame.shape = (frame.shape[0], 1)
        group.create_dataset('Frame Number', (frame.shape[0], frame.shape[1]), dtype='int16', data=frame)
        del data

        # data = (f.select('Infrared Channel Calibration Data')).get()
        # group.create_dataset('Infrared Channel Calibration Data', (data.shape[0], data.shape[1], data.shape[2]),
        #                      dtype=type(data[0][0][0]), data=data)
        # del data

        data = f['Millisecond/Millisecond'][()]
        millisecond = group.create_dataset('Millisecond', (data.shape[0], data.shape[1]), dtype=np.float64, data=data)
        millisecond.attrs['Fillin_value'] = -999.0
        millisecond.attrs['Unit'] = np.string_('milliseconds since at 00:00:00 on this day ')
        del data

        # data = (f.select('Mirror-side Flag')).get()
        #     # group.create_dataset('Mirror-side Flag', (data.shape[0], data.shape[1]), dtype='|S1', data=data)
        #     # del data
        # no ORB_VEC dataset in HDF
        # data = (f.select('ORB_VEC')).get()
        # group.create_dataset('ORB_VEC', (data.shape[0], data.shape[1]), dtype=type(data[0][0]), data=data)
        # del data
        group.attrs['Instrument Parameters'] = np.string_('412nm,443nm,490nm,520nm,565nm,670nm,750nm,865nm,10.3-11.4um,'
                                                          '11.5-12.5um')

        #         file attributes

        f_new.attrs.create('Calibration Flag', 'None', shape=(1,), dtype='S10')
        f_new.attrs.create('Calibration Version', '1.00', shape=(1,), dtype='S10')
        f_new.attrs.create('DayorNight', 'D', shape=(1,), dtype='S10')
        distance = 1 - 0.01672 * np.cos(0.9856 * (int(os.path.basename(infile)[8:11]) - 4))
        f_new.attrs.create('Earth-Sun Distance', distance, shape=(1,), dtype=np.float32)
        f_new.attrs.create('Easternmost Longitude', np.max(lon), shape=(1,), dtype=np.float32)
        f_new.attrs.create('End Center Longitude', center_lon[-1], shape=(1,), dtype=np.float32)
        f_new.attrs.create('End Center Latitude', center_lat[-1], shape=(1,), dtype=np.float32)
        f_new.attrs.create('GEO Correction Method', 'Unkonwn', shape=(1,), dtype='S10')
        f_new.attrs.create('Input File', os.path.basename(infile), shape=(1,), dtype='S10')
        f_new.attrs.create('Latitude Unit', 'degree', shape=(1,), dtype='S10')
        f_new.attrs.create('Longitude Unit', 'degree', shape=(1,), dtype='S10')
        f_new.attrs.create('Lower Left Latitude', lat[-1, 0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Lower Left Longitude', lon[-1, 0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Lower Right Latitude', lat[-1, -1], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Lower Right Longitude', lon[-1, -1], shape=(1,), dtype=np.float32)

        node = root[0].find("NodeCrossingTime")
        time = datetime.datetime(int('20' + node.text[0:2]), 1, 1) + datetime.timedelta(days=int(node.text[2:5]) - 1) + \
               datetime.timedelta(hours=int(node.text[5:7])) + datetime.timedelta(minutes=int(node.text[7:9]))
        f_new.attrs.create('Node Crossing Time', time.strftime("%Y-%m-%dT%H-%M-%S"), shape=(1,), dtype='S10')
        f_new.attrs.create('Northernmost Latitude', np.max(lat), shape=(1,), dtype=np.float32)
        f_new.attrs.create('Number of Bands', 16, shape=(1,), dtype=np.int16)
        f_new.attrs.create('Number of Scan Lines', f['Scan image data/L_412'].shape[0], shape=(1,), dtype=np.int32)
        f_new.attrs.create('Orbit Node Longitude', root[0].find("OrbitNodeLongitude").text, shape=(1,),
                           dtype=np.float32)
        f_new.attrs.create('Orbit Number', root[0].find("OrbitNumber").text, shape=(1,), dtype=np.int32)
        f_new.attrs.create('Pixels Per Scan Line', 1024, shape=(1,), dtype=np.int32)
        f_new.attrs.create('Processing Center', 'NSOAS', shape=(1,), dtype='S10')
        f_new.attrs.create('Processing Control', 'IMG', shape=(1,), dtype='S10')
        f_new.attrs.create('Processing Time', datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), shape=(1,),
                           dtype='S10')
        f_new.attrs.create('Product Name', os.path.basename(outfile), shape=(1,), dtype='S10')
        f_new.attrs.create('Radiometric Method', 'unknown', shape=(1,), dtype='S10')

        day = datetime.datetime(int('20' + os.path.basename(infile)[6:8]), 1, 1) + \
              datetime.timedelta(days=int(os.path.basename(infile)[8:11]) - 1)
        f_new.attrs.create('Range Beginning Date', day.strftime("%Y%m%d") + ' ' + os.path.basename(infile)[8:11],
                           shape=(1,),
                           dtype='S10')
        f_new.attrs.create('Range Ending Date', day.strftime("%Y%m%d") + ' ' + os.path.basename(infile)[8:11],
                           shape=(1,),
                           dtype='S10')
        hour = np.min(millisecond) % (1000 * 60 * 60)
        minute = (np.min(millisecond) - hour * (1000 * 60 * 60)) % (1000 * 60)
        second = (np.min(millisecond) - hour * (1000 * 60 * 60) - minute * 1000 * 60) / (1000)
        f_new.attrs.create('Range Beginning Time', str(hour) + ':' + str(minute) + ':' + str(second), shape=(1,),
                           dtype='S10')

        hour = np.max(millisecond) // (1000 * 60 * 60)
        minute = (np.max(millisecond) - hour * (1000 * 60 * 60)) // (1000 * 60)
        second = (np.max(millisecond) - hour * (1000 * 60 * 60) - minute * 1000 * 60) // (1000)
        f_new.attrs.create('Range Ending Time', str(hour) + ':' + str(minute) + ':' + str(second), shape=(1,),
                           dtype='S10')
        f_new.attrs.create('Realtime Delay Flag', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Receiving End Time', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Receiving Start Time', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Ref Band Number', 6, shape=(1,), dtype=np.int32)

        node = root[0].find("MissionCharacter")
        f_new.attrs.create('Satellite Character', node.text, shape=(1,), dtype='S10')
        f_new.attrs.create('Satellite Name', 'HY-1A', shape=(1,), dtype='S10')
        node = root[0].find("SceneCenterLatitude")
        f_new.attrs.create('Sence Center Latitude', node.text, shape=(1,), dtype=np.float32)
        node = root[0].find("SceneCenterLongitude")
        f_new.attrs.create('Sence Center Longitude', node.text, shape=(1,), dtype=np.float32)
        node = root[0].find("SceneCenterSolarZenith")
        f_new.attrs.create('Sence Center Solar Zenith', node.text, shape=(1,), dtype=np.float32)
        node = root[0].find("SceneCenterTime")
        time = datetime.datetime(int('20' + node.text[0:2]), 1, 1) + datetime.timedelta(days=int(node.text[2:5]) - 1) + \
               datetime.timedelta(hours=int(node.text[5:7])) + datetime.timedelta(minutes=int(node.text[7:9]))
        f_new.attrs.create('Sence Center Solar Time', time.strftime("%Y-%m-%dT%H-%M-%S"), shape=(1,), dtype='S10')
        f_new.attrs.create('Sensor Mode', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Sensor Name', 'COCTS, Chinese Ocean Color and Temperature Scanner', shape=(1,), dtype='S10')
        f_new.attrs.create('Sensor Pitch Element', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Sensor Yaw Element', 'Unknown', shape=(1,), dtype='S10')
        f_new.attrs.create('Software Version', '01.00', shape=(1,), dtype='S10')
        f_new.attrs.create('Southernmost Latitude', np.min(lat), shape=(1,), dtype=np.float32)
        f_new.attrs.create('Start Center Longitude', center_lon[0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Start Center Latitude', center_lat[0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('TLE', '', shape=(1,), dtype='S10')
        f_new.attrs.create('The Parameters of Sensor Characteristics',
                           '412nm,443nm,490nm,520nm,565nm,670nm,750nm,865nm,10.3-11.4um,'
                           '11.5-12.5um', shape=(1,), dtype='S10')
        f_new.attrs.create('Title', 'HY-1A OCT Level-1B', shape=(1,), dtype='S10')
        f_new.attrs.create('Upper Left Latitude', lat[0, 0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Upper Left Longitude', lon[0, 0], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Upper Right Latitude', lat[0, -1], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Upper Right Longitude', lon[0, -1], shape=(1,), dtype=np.float32)
        f_new.attrs.create('Westernmost Longitude', np.min(lon), shape=(1,), dtype=np.float32)
        f.close()
        f_new.close()
        return outfile


if __name__ == '__main__':

    # HY1A L1B数据说明文件
    HY1AL1Bfile_illustrate_file = r"F:\DATA\HY1A\L1B\allData/HY1AL1Bfile_illustrate.xlsx"

    # 将0级数和L1B级数据放在同一个文件夹下
    filedir = r'F:\DATA\HY1A\L1B\allData\parsed'
    files = glob.glob(filedir + os.sep + "*L1B.bz2")
    # files.append(glob.glob(filedir + "/*BZ2"))
    pbar = tqdm(total=len(files), desc='processing:')
    for index, file in enumerate(files):
        pbar.update(1)
        # print(file)
        # if index <200:
        #    continue
        try:
            L1B = L1bparase()
            L1bfile = L1B.read_write(file, HY1AL1Bfile_illustrate_file)
            if L1bfile == 'error':
                print(os.path.basename(file)+'--data error,continue...')
                continue

            # print(L1bfile)
            to1c = to1cformat()
            outfile = to1c.read2h5(L1bfile)
            #  os.remove(L1bfile)
        except:
            #  shutil.move(file,r'H:\DATA\HY1A\L1B\allData\parsed\wrong'+os.sep+os.path.basename(file))
           continue

        # gc.collect()
    pbar.close()

