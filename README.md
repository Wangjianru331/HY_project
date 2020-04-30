# 这是一个公开目录
 
## 2020.4.28
### 针对HY1B部分数据定标系数offset的严重错误（值为负几百万）修改了程序HY1B_TOA_nreflectance_timeseries.py:
    if offset[i]<-100 or offset[i]>100:
                offset_=0
