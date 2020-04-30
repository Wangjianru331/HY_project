# 这是一个公开目录
 ## 程序说明
 ### 1 MODIS_vzalt5_pixel_information_allData.py：用于从单景影像对应的csv文件中筛选MODIS vza小于5度的像元数据，并输出为一个总的CSV文件
 ### 2 distributiondensity_MODIS_vzaLT5.py：绘制MODIS vza小于5度的像元的空间分布密度
 ## 2020.4.28
### 针对HY1B部分数据定标系数offset的严重错误（值为负几百万）修改了程序HY1B_TOA_nreflectance_timeseries.py:
    if offset[i]<-100 or offset[i]>100:
                offset_=0
 ## 2020.4.30
 
