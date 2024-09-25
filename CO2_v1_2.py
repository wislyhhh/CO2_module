from pyhdf.SD import SD
from netCDF4 import Dataset
import math
import proplot as pplt
import pandas as pd
import xarray as xr
import numpy as np
from datetime import date
import configparser
import warnings
import os
import sys
# def find_nearest(array0, array1):
#     len_arr1 = array1.shape[0]
#     idx_f = np.full(len_arr1, np.nan)
#     for i in np.arange(len_arr1):
#         value = array1[i]
#         array0 = np.asarray(array0)
#         idx_f[i] = (np.abs(array0 - value)).argmin()
#     return idx_f

# def slopeVar(array):
#     S2 = np.full([np.size(array, 0), np.size(array, 1)], np.nan)
#     for i in np.arange(np.size(array, 0)):
#         for j in np.arange(np.size(array, 1)):
#             temp = array[i,j]
#             if 1<=temp<=7:
#                 S2[i,j] = (math.log(temp) + 1.2)*(1e-2)
#             else: 
#                 if temp>7:
#                     S2[i,j] = (0.85*math.log(temp) - 1.45)*(1e-1)
#                 else:
#                     S2[i,j] = (math.log(1) + 1.2)*(1e-2)
#     return S2  
#2.0再更新向量化的代码
def find_nearest(array0, array1):
    array0 = np.asarray(array0)
    
    if np.isscalar(array1):
        idx_f = (np.abs(array0 - array1)).argmin()
    else:
        idx_f = np.full(len(array1), np.nan)

        if len(array1.shape)>1:
            idx_f = (np.abs(array0 - array1)).argmin(axis=1)
        else:
            idx_f = (np.abs(array0 - array1[:, None])).argmin(axis=1)

    return idx_f

def slopeVar(array):
    # 计算array的维度
    Len_Lat = array.shape[0]
    Len_Long = array.shape[1]

    # 创建一个用于存储斜率变化值的数组
    S2 = np.full((Len_Lat, Len_Long), np.nan)

    # 计算斜率变化值
    S2[(1<=array) & (array<=7)] = (np.log(array[(1<=array) & (array<=7)]) + 1.2) * (1e-2)
    S2[array>7] = (0.85*np.log(array[array>7]) - 1.45) * (1e-1)
    S2[array<=0] = (np.log(1) + 1.2) * (1e-2)

    return S2

def nanMovAverage(Array_AfterMA, Array_ori, windowSize):    
    Len_Lat_AfterMA = Array_AfterMA.shape[0]
    Len_Long_AfterMA = Array_AfterMA.shape[1]
    flag=[True,True,True]
    for i in np.arange(Len_Lat_AfterMA):
        for j in np.arange(Len_Long_AfterMA):
            Array_window =  Array_ori[(windowSize*i):(windowSize*i+windowSize),(windowSize*j):(windowSize*j+windowSize)]
            Array_AfterMA[i,j] = np.nanmean(Array_window)
        if abs(i/Len_Lat_AfterMA-0.3)<1e-2 and flag[0]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成30%\n') 
            print('MODIS和CARE地表数据预处理完成20%')
            sys.stdout.flush()
            flag[0]=False
        if abs(i/Len_Lat_AfterMA-0.6)<1e-2 and flag[1]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成60%\n') 
            print('MODIS和CARE地表数据预处理完成40%')
            sys.stdout.flush()
            flag[1]=False
        if abs(i/Len_Lat_AfterMA-0.9)<1e-2 and flag[2]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成90%\n') 
            print('MODIS和CARE地表数据预处理完成60%')
            sys.stdout.flush()
            flag[2]=False
        
    return Array_AfterMA 

def read_txt(file_path):
    """读取 .txt 文件并返回数据框"""
    df = pd.read_csv(file_path, delimiter='\t')  # 假设数据以制表符分隔
    return df

def read_excel(file_path):
    """读取 .xlsx 文件并返回数据框"""
    df = pd.read_excel(file_path)
    return df

def extract_data(df,year_int, month_int, day_int):
    """从数据框中提取数据并返回字典"""
    
    dateNum_CAL=date.toordinal(date(year_int, month_int, day_int)) + 366
    Lat_CAL = df['Latitude'].tolist()
    Lon_CAL = df['Longitude'].tolist()
    Surf_Elvation = df['Surface_Elevation'].tolist()
    Day_Night_Flag = df['Day_Night_Flag'].tolist()
    AOD_S532 = df['AOD_S532'].tolist()
    AOD_T532 = df['AOD_T532'].tolist()
    COD_532 = df['COD_532'].tolist()

    Data_CAL_dict = {
        'dateNum_CAL': [dateNum_CAL] * len(df),
        'Lat_CAL': Lat_CAL,
        'Lon_CAL': Lon_CAL,
        'Surf_Elvation': Surf_Elvation,
        'Day_Night_Flag': Day_Night_Flag,
        'AOD_S532': AOD_S532,
        'AOD_T532': AOD_T532,
        'COD_532': COD_532
    }
    return Data_CAL_dict

def convert_to_dataframe(data_dict):
    """将字典转换为 pandas 数据框"""
    df = pd.DataFrame(data_dict)
    return df

def fusion(inputpath,Path):
    warnings.filterwarnings("ignore")
    # 从GUI中读取变量
    config = configparser.ConfigParser()                                           # 启动ConfigParse
    outputpath=Path+'/output_fusion/'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    config.read(Path+'/Config_Parameters.ini')                                     # 使用read()读取文件
    day_night_flag= config['NEWSETTINGS'].getboolean('day_night_flag')
    CAL_fileName = config['NEWSETTINGS']['CAL_fileName']
    config.read(inputpath+'/Config_Filenames.ini')                                     # 使用read()读取文件
    CO2_IS_fileName = config['NEWSETTINGS']['CO2_IS_fileName']
    MODIS_LR_fileName = config['NEWSETTINGS']['MODIS_LR_fileName']
    CARE_SR_fileName = config['NEWSETTINGS']['CARE_SR_fileName']
    LS_flag_fileName = config['NEWSETTINGS']['LS_flag_fileName']
    ATM_fileName = config['NEWSETTINGS']['ATM_fileName']
        
    # 定义物理常量或者不变量
    Rf = 0.08;                                                                     # 1.5um下的海沫（whitecaps）等效反射率
    Rou = 0.02;                                                                    # 菲涅尔等效反射系数
    Ru = 0;                                                                        # 考虑1.57um水下的后向散射为0(>0.7um)
    g = 9.8066                                                                     # 重力加速度
    #%% CALIPSO数据预处理
    # start = time.time()
    year_int = int(CAL_fileName[35:39])
    month_int = int(CAL_fileName[40:42])
    day_int = int(CAL_fileName[43:45])
    file_ext = os.path.splitext(CAL_fileName)[1]
    file_path = os.path.join(inputpath, 'CAL', CAL_fileName)
    # 数据读取
     
    if file_ext == '.hdf':
     
     
     CAL_file = SD(file_path)
     print("hdf file read successfully.")
     # CAL_datasets = CAL_file.datasets() 
     Data_CAL_dict = {"dateNum_CAL": date.toordinal(date(year_int, month_int, day_int)) + 366, 
                    "Lat_CAL": CAL_file.select('Latitude')[:, 2].tolist(),                            # IPDA正反演过程主要围绕CALIPSO的经纬度和时间展开
                    "Lon_CAL": CAL_file.select('Longitude')[:, 2].tolist(),
                    "Surf_Elvation": CAL_file.select('DEM_Surface_Elevation')[:, 3].tolist(),         # 确定地表起伏
                    "Day_Night_Flag": CAL_file.select('Day_Night_Flag')[:].reshape(-1,).tolist(), 
                    "AOD_S532": CAL_file.select('Column_Optical_Depth_Stratospheric_Aerosols_532')[:].reshape(-1,).tolist(), 
                    "AOD_T532": CAL_file.select('Column_Optical_Depth_Tropospheric_Aerosols_532')[:].reshape(-1,).tolist(), 
                    "COD_532": CAL_file.select('Column_Optical_Depth_Cloud_532')[:].reshape(-1,).tolist()}
     Data_CAL = pd.DataFrame(Data_CAL_dict)
    elif file_ext == '.xlsx':
        df = read_excel(file_path)
        print("Excel file read successfully.")
        data_dict = extract_data(df,year_int, month_int, day_int)
        Data_CAL = convert_to_dataframe(data_dict)
    elif file_ext == '.txt':
        df = read_txt(file_path) 
        data_dict = extract_data(df,year_int, month_int, day_int)
        Data_CAL = convert_to_dataframe(data_dict)
    else:
        raise ValueError("Unsupported file format: " + file_ext) 
     


    # 确定数据有效范围(官方定义)
    Data_CAL.loc[(Data_CAL["Surf_Elvation"]<-1) | 
                (Data_CAL["Surf_Elvation"]>9), "Surf_Elvation"] = np.nan          # 使用DafaFrameming.loc[行名, 列名]=值的方式去赋值, 而不是使用DataFrame[][]的形式去赋值
    Data_CAL.loc[(Data_CAL["AOD_S532"]<0) | 
                (Data_CAL["AOD_S532"]>3), "AOD_S532"] = np.nan
    Data_CAL.loc[(Data_CAL["AOD_T532"]<0) | 
                (Data_CAL["AOD_T532"]>3), "AOD_T532"] = np.nan
    Data_CAL.loc[(Data_CAL["COD_532"]<0) | 
                (Data_CAL["COD_532"]>25), "COD_532"] = np.nan
    Data_CAL["AOD_532"] = Data_CAL["AOD_S532"] + Data_CAL["AOD_T532"]
    Data_CAL = Data_CAL.drop(columns = ["AOD_S532", "AOD_T532"])

    # 确定D/N
    # Data_CAL = Data_CAL[Data_CAL["Day_Night_Flag"]==day_night_flag]                # DataFrame[]默认对行进行处理，Data_CAL["Day_Night_Flag"]==day_night_flag是存有布尔类型值的series
                                                                                # AAA = np.array([[1,2,3],[4,5,6],[7,8,9]])    # 输入[[1,2,3],[4,5,6],[7,8,9]]是一个list
                                                                                # BBB = AAA[0:2]                               # ndarray[]默认对行进行处理
    # 去除足迹点云层过厚的点
    TOD0 = np.hstack((Data_CAL["AOD_532"].values.reshape(-1, 1), Data_CAL["COD_532"].values.reshape(-1, 1)))
    Data_CAL["TOD"] = np.nansum(TOD0, axis=1).tolist()
    Data_CAL = Data_CAL[~((Data_CAL["TOD"] > 1) | (np.isnan(Data_CAL["TOD"])))]    # 记得要对SQE2取反，否则就与MATLAB版本冲突了

    # 单位转换
    Data_CAL["Surf_Elvation"] = Data_CAL["Surf_Elvation"]*1000                     # 单位转换放到最后进行，否则会影响有效范围确定                                          

    # 数据保存
    Data_CAL.to_csv(outputpath+"ProcessedData_CALIPSO.csv", index=False)
    print('CALIPSO光学厚度预处理')
    
    sys.stdout.flush()
    #%% CO2先验值数据预处理
    # start = time.time()
    
    # 数据读取及预处理
    year_int = int(CO2_IS_fileName[14:18])
    month_int = int(CO2_IS_fileName[18:20])
    CO2_IS_fileName_full = inputpath + CO2_IS_fileName
    NC_file = Dataset(CO2_IS_fileName_full)
    Cams_datenum0 = date.toordinal(date(year_int, month_int, 1)) + 366
    time_cams = NC_file.variables['time'][:].reshape(-1,)
    Cams_datenum = np.floor(Cams_datenum0 + time_cams/24)
    XCO2 = NC_file.variables['XCO2'][:]
    Lat_cams = NC_file.variables['latitude'][:].reshape(-1,)
    Long_cams = NC_file.variables['longitude'][:].reshape(-1,)
    SQE3 = Cams_datenum == Data_CAL["dateNum_CAL"].values[0]
    Pos = np.where(SQE3)[0]
    if day_night_flag==0:
        pos = Pos[4]                                                               # ？？？为啥要这么取
    else:
        pos = Pos[0]
    xco2 = XCO2[pos,:,:]

    # CO2先验值与CAL数据融合
    xco2_CAL = np.full(Data_CAL.shape[0], np.nan)
    Idx_lat = find_nearest(Lat_cams, Data_CAL["Lat_CAL"].values).astype(np.int32)
    Idx_long = find_nearest(Long_cams, Data_CAL["Lon_CAL"].values).astype(np.int32)
    xco2_CAL = xco2[Idx_lat, Idx_long]
    xco2_CAL = xco2_CAL*(1e-6)                                                     # 单位转换
    Data_xco2_CAL_dict = {"dateNum_CAL": Data_CAL["dateNum_CAL"].tolist(), 
                        "Lat_CAL": Data_CAL["Lat_CAL"].tolist(), 
                        "Lon_CAL": Data_CAL["Lon_CAL"].tolist(), 
                        "xco2_CAL": xco2_CAL.tolist()}                                          
    Data_xco2_CAL = pd.DataFrame(Data_xco2_CAL_dict)

    # 数据保存
    Data_xco2_CAL.to_csv(outputpath+"ProcessedData_xco2_CAL.csv", index=False)
    print('ERA5-XCO2初始场预处理完成')
    sys.stdout.flush()
        
#%% MODIS和CARE数据预处理
    # CARE
    CARE_SR_fileName_full = inputpath + CARE_SR_fileName
    ds_CARE = xr.open_dataset(CARE_SR_fileName_full)
    time_CARE = ds_CARE["time"].values.astype("datetime64[D]").astype(str)
    Latitude_CARE = ds_CARE["latitude"].values
    Longitude_CARE = ds_CARE["longitude"].values
    u10_CARE = ds_CARE["u10"].values
    ds_CARE.close()                                                                # 打开后记得关闭
    SQE4 = time_CARE == CAL_fileName[35:45]
    Pos = np.where(SQE4)[0]
    if day_night_flag==0:
        pos = Pos[1]
    else:
        pos = Pos[0]
    u10_CARE = u10_CARE[pos]
    u10_CARE[u10_CARE==-32767] = np.nan
    u10_CARE = u10_CARE*9.1130e-04-2.5004
    u10_CARE = np.abs(u10_CARE)

    W = 2.95*(1E-6)*pow(u10_CARE, 3.52)
    S2 = slopeVar(u10_CARE)
    Rs = Rou/(4*S2)
    Rsea = (1-W)*Rs + W*Rf + (1 - W*Rf)*Ru

    # 2.0更新MODIS的向量化读取，改用xarray
    # MODIS
    MODIS_LR_fileName_full = inputpath + MODIS_LR_fileName
    MODIS_file = SD(MODIS_LR_fileName_full)
    Reflect_Band6 = MODIS_file.select('Nadir_Reflectance_Band6')[:].astype(float)  # 不转换为float的话，nan无法赋进去                               
    BRDF_Quality = MODIS_file.select('BRDF_Quality')[:]
    Reflect_Band6[(Reflect_Band6==32767) | (BRDF_Quality==255) | (BRDF_Quality==6) | (BRDF_Quality==7)] = np.nan
    Reflect_Band6 = Reflect_Band6*0.0001
    Rland = np.full([Latitude_CARE.shape[0], Longitude_CARE.shape[0]], np.nan)                   # Rland的初始化
    print('MODIS和CARE地表数据已读取')
    sys.stdout.flush()
    Rland = nanMovAverage(Rland, Reflect_Band6, 10)

    # Land_flag
    LS_flag_fileName_full = inputpath + LS_flag_fileName
    ds_lf = xr.open_dataset(LS_flag_fileName_full)
    Land_flag = ds_lf["Land_flag"].values
    ds_lf.close()

    # MODIS、CARE与CAL数据融合
    NadirReflect = np.full([len(Latitude_CARE), len(Longitude_CARE)], np.nan)
    for i in np.arange(len(Latitude_CARE)):
        for j in np.arange(len(Longitude_CARE)):
            rland = Rland[i, j]
            rsea = Rsea[i, j]
            land_flag = Land_flag[i, j]
            if ~np.isnan(rland):
                NadirReflect[i, j] = rland
            else:
                if land_flag == 1:
                    NadirReflect[i, j] = 0.1
                else:
                    NadirReflect[i, j] = rsea
    print('MODIS和CARE地表数据预处理完成80%')
    sys.stdout.flush()
    rland = Rland
    rsea = Rsea
    land_flag = Land_flag
    # 2.0 版本再更新
    # NadirReflect = np.where(~np.isnan(rland), rland, np.where(land_flag == 1, 0.1, rsea))
    # NadirReflect_CAL = np.full(len(Data_CAL["Lat_CAL"]), np.nan)
    Idx_lat = find_nearest(Latitude_CARE, Data_CAL["Lat_CAL"].values).astype(np.int32)
    Idx_long = find_nearest(Longitude_CARE, Data_CAL["Lon_CAL"].values).astype(np.int32)
    NadirReflect_CAL = NadirReflect[Idx_lat, Idx_long]
    print('MODIS和CARE地表数据预处理完成100%')
    sys.stdout.flush()
    Data_MODISandCARE_CAL_dict = {"dateNum_CAL": Data_CAL["dateNum_CAL"].tolist(), 
                                "Lat_CAL": Data_CAL["Lat_CAL"].tolist(), 
                                "Lon_CAL": Data_CAL["Lon_CAL"].tolist(),
                                "NadirReflect_CAL": NadirReflect_CAL.tolist()}
    Data_MODISandCARE_CAL = pd.DataFrame(Data_MODISandCARE_CAL_dict)

    Data_MODISandCARE_CAL.to_csv(outputpath+"ProcessedData_MODISandCARE_CAL.csv")
    print('MODIS和CARE地表数据预处理结果已保存')
    sys.stdout.flush()
    # print('MODIS和CARE数据预处理运行时间：', runtime) 
    sys.stdout.flush()
    #%% 大气数据预处理
    # start = time.time()
    
    # 数据读取
    ATM_fileName_full = inputpath + ATM_fileName
    ds_ATM = xr.open_dataset(ATM_fileName_full)
    SH = ds_ATM['q'].values[0,:,:,:]
    Tem = ds_ATM['t'].values[0,:,:,:]
    Pre = ds_ATM["level"].values*100
    Height = ds_ATM["z"].values[0,:,:,:]/g
    Lat_Atom = ds_ATM["latitude"].values
    Long_Atom = ds_ATM["longitude"].values
    ds_ATM.close()

    SH_CAL = np.full([np.size(Data_CAL["Lat_CAL"]), np.size(SH, 0)], np.nan)
    Tem_CAL = np.full([np.size(Data_CAL["Lat_CAL"]), np.size(Tem, 0)], np.nan)
    Height_CAL = np.full([np.size(Data_CAL["Lat_CAL"]), np.size(Height, 0)], np.nan)
    for i in np.arange(len(Data_CAL["Lat_CAL"])):
        lat_CAL = Data_CAL["Lat_CAL"].values[i].reshape(-1, 1)
        Lon_CAL = Data_CAL["Lon_CAL"].values[i].reshape(-1, 1)
        idx_lat = find_nearest(Lat_Atom, lat_CAL).astype(np.int32)[0]
        idx_long = find_nearest(Long_Atom, Lon_CAL).astype(np.int32)[0]
        SH_CAL[i,:] = SH[:, idx_lat, idx_long]                                     # 等号左右两边维度要相等
        Tem_CAL[i,:] = Tem[:, idx_lat, idx_long]
        Height_CAL[i,:] = Height[:, idx_lat, idx_long]

    Data_Atomsphere_CAL = xr.Dataset({"Height": (("Lat_CAL", "Pressure"), Height_CAL), 
                                    "Specific_Humidity": (("Lat_CAL", "Pressure"), SH_CAL), 
                                    "Temperature": (("Lat_CAL", "Pressure"), Tem_CAL)}, 
                                    coords={"Lat_CAL": Data_CAL["Lat_CAL"].values, 
                                            "Pressure": Pre})

    Data_Atomsphere_CAL.to_netcdf(outputpath+"ProcessedData_Atmosphere_CAL.nc")
    print('大气温湿压数据融合完成')
    sys.stdout.flush()
    print("如您对程序结果有任何问题,请联系")
    sys.stdout.flush()
    print("浙江大学刘东教授（liudongopt@zju.edu.cn）团队")
    sys.stdout.flush()

def simulation(inputpath,Path):
    # 从GUI中读取变量
    config = configparser.ConfigParser()  
    # config.optionxform= lambda option:option                              # 启动ConfigParse
    # inputpath=Path+'/input/'
    fusionpath=Path+'/output_fusion/'
    outputpath=Path+'/output_simulation/'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    config.read(inputpath+'/Config_FileNames.ini')                           # 使用read()读取文件
    XS_LUT_fileName=config['NEWSETTINGS']['XS_LUT_fileName']
    config.read(Path+'/Config_Parameters.ini')                           # 使用read()读取文件
    working_wavelength = config['NEWSETTINGS'].getfloat('working_wavelength')       
    reference_wavelength= config['NEWSETTINGS'].getfloat('reference_wavelength')
    energy_on= config['NEWSETTINGS'].getfloat('energy_on')*(1e-3)
    energy_off= config['NEWSETTINGS'].getfloat('energy_off')*(1e-3)
    optical_efficient= config['NEWSETTINGS'].getfloat('optical_efficient')
    satellite_altitude= config['NEWSETTINGS'].getfloat('satellite_altitude')*(1e3)
    pulse_width= config['NEWSETTINGS'].getfloat('pulse_width')*(1e-19)  
    # repetition_rate= config['NEWSETTINGS'].getfloat('repetition_rate')
    FOV= config['NEWSETTINGS'].getfloat('FOV')
    telescope_diam= config['NEWSETTINGS'].getfloat('telescope_diam')
    filter_bandwidth= config['NEWSETTINGS'].getfloat('filter_bandwidth')*(1e+6) 
    responsivity= config['NEWSETTINGS'].getfloat('responsivity')
    excess_noise_factor= config['NEWSETTINGS'].getfloat('excess_noise_factor')
    intergal_G_factor= config['NEWSETTINGS'].getfloat('intergal_G_factor')
    day_night_flag= config['NEWSETTINGS'].getboolean('day_night_flag')
    

    # 定义物理常量或者不变量
    NA = 6.02214129*(1e+23)               # 阿伏伽德罗常数 1/mol
    R = 8.31441                           # 普适气体常量J/(mol·k)
    PI = 3.1415926
    O = 1                                 # 重叠因子
    c = 2.99792458*(1e+8)                 # 光速，单位m/s
    OD_AIR = 0
    g = 9.8066                            # 重力加速度
    Mair = 28.964425278793996*(1e-3)      # 空气平均摩尔质量，单位kg/mol
    MH2O = 18*(1e-3)                      # 水的平均摩尔质量，单位kg/mol
    e = 1.6021892*(1e-19)                 # 电子电量 C
    I_dark = 64*(1e-15)                   # 暗电流/NEP w/sqrt(Hz)
    RL = 1*(1e+6)                         # 反馈电阻欧姆
    Kb = 1.3806488*(1e-23)                # 玻尔兹曼常数J/K            
    T = 296                               # 探测器的工作温度K   4*Kb*T./RL
    Nshots = 148
    
    # 数据预处理
    A = PI*pow((telescope_diam/2), 2)
    optical_efficient = optical_efficient/3.1415                                   # 总的光学系统透过率

    #%% 正演模拟仿真
    
    # 读取查找表数据
    XS_LUT_fileName_full = inputpath + XS_LUT_fileName
    ds_XS = xr.open_dataset(XS_LUT_fileName_full)
    Pre_LUT = ds_XS["Pressure"].values
    Tem_LUT = ds_XS["Temperature"].values
    CO2_FAbsON = ds_XS["XS_LUT_CO2_ON"].values
    CO2_FAbsOFF = ds_XS["XS_LUT_CO2_OFF"].values
    H2O_FAbsON = ds_XS["XS_LUT_H2O_ON"].values
    H2O_FAbsOFF = ds_XS["XS_LUT_H2O_OFF"].values
    ds_XS.close()
    # 读取CALIPSO数据
    CALIPSO_fileName_full = fusionpath + "ProcessedData_CALIPSO.csv"
    Data_CAL=pd.read_csv(CALIPSO_fileName_full)
    # 读取MCC融合数据
    MCC_fileName_full = fusionpath + "ProcessedData_MODISandCARE_CAL.csv"
    Data_MCC=pd.read_csv(MCC_fileName_full)
    NadirReflect_CAL=Data_MCC["NadirReflect_CAL"]
    # 读取XCO2初始场
    XCO2_filename_full=fusionpath + "ProcessedData_xco2_CAL.csv"
    Data_XCO2=pd.read_csv(XCO2_filename_full)
    xco2_CAL=Data_XCO2["xco2_CAL"]
    # 读取大气数据
    ATM_fileName_full = fusionpath + "ProcessedData_Atmosphere_CAL.nc"
    Data_ATM=xr.open_dataset(ATM_fileName_full)
    Tem_CAL=Data_ATM["Temperature"].values
    Height_CAL=Data_ATM["Height"].values
    SH_CAL=Data_ATM["Specific_Humidity"].values
    # Lat_CAL=Data_ATM.coords["Lat_CAL"].values
    Pre=Data_ATM.coords["Pressure"].values
    
    Data_ATM.close()
    print('数据读取已完成')
    sys.stdout.flush()
    
    # 主体代码
    v = np.array([working_wavelength, reference_wavelength])
    if day_night_flag == 0:
        R_solar = 10*(1e-3)
    else:
        R_solar = 0
    P_ON = np.full(len(Data_CAL["Lat_CAL"]), np.nan).reshape(-1,1)
    P_OFF = np.full(len(Data_CAL["Lat_CAL"]), np.nan).reshape(-1,1)
    DAOD_CO2_IPDA = np.full(len(Data_CAL["Lat_CAL"]), np.nan).reshape(-1,1)
    flag=[True,True,True]
    for i in np.arange(len(Data_CAL["Lat_CAL"])):
        ref = NadirReflect_CAL[i]
        surf_std = Data_CAL["Surf_Elvation"].values[i]
        tod = Data_CAL["TOD"].values[i]
        tem = Tem_CAL[i, :].reshape(-1, 1)                                         # tem、sh、height和Pre保持统一就好了，最大往小积分和最小往大积分结果一样
        sh = SH_CAL[i, :].reshape(-1, 1)
        height = Height_CAL[i, :].reshape(-1, 1)
        xco2_0 = xco2_CAL[i]
        
        delt_h = surf_std                                                          # 激光足迹内的高度变化，单位m
        Delt_teff = math.sqrt(pow(pulse_width, 2) + pow(1/(3*filter_bandwidth), 2) + pow(2*delt_h/c, 2))
        od = tod + OD_AIR
        
        Abs_CO2_on = np.full(np.size(tem), np.nan).reshape(-1,1)
        Abs_CO2_off = np.full(np.size(tem), np.nan).reshape(-1,1)
        Abs_H2O_on = np.full(np.size(tem), np.nan).reshape(-1,1)
        Abs_H2O_off = np.full(np.size(tem), np.nan).reshape(-1,1)
        
        Idx_tem = find_nearest(Tem_LUT, tem).astype(np.int32)
        Idx_pre = find_nearest(Pre_LUT, Pre).astype(np.int32)
        Abs_CO2_on = CO2_FAbsON[Idx_tem, Idx_pre].reshape(-1, 1)
        Abs_CO2_off = CO2_FAbsOFF[Idx_tem, Idx_pre].reshape(-1, 1)
        Abs_H2O_on = H2O_FAbsON[Idx_tem, Idx_pre].reshape(-1, 1)
        Abs_H2O_off = H2O_FAbsOFF[Idx_tem, Idx_pre].reshape(-1, 1)
        Abs_CO2 = np.hstack((Abs_CO2_on, Abs_CO2_off))
        Abs_H2O = np.hstack((Abs_H2O_on, Abs_H2O_off))
        
        if i==0:
            Abs_CO2_ON_ALL = Abs_CO2_on
            Abs_CO2_OFF_ALL = Abs_CO2_off
            Abs_H2O_ON_ALL = Abs_H2O_on
            Abs_H2O_OFF_ALL = Abs_H2O_off
        else:
            Abs_CO2_ON_ALL = np.hstack((Abs_CO2_ON_ALL, Abs_CO2_on))
            Abs_CO2_OFF_ALL = np.hstack((Abs_CO2_OFF_ALL, Abs_CO2_off))
            Abs_H2O_ON_ALL = np.hstack((Abs_H2O_ON_ALL, Abs_H2O_on))
            Abs_H2O_OFF_ALL = np.hstack((Abs_H2O_OFF_ALL, Abs_H2O_off))
        
        # 测试：维度不同的矩阵之间乘除计算
        # AAAA = np.array([[1,2],[3,4], [5,6]])
        # BBBB = np.array([1,2,3]).reshape(-1,1)
        # CCCC0 = AAAA/BBBB
        # CCCC1 = AAAA*BBBB
        OD_CO2 = np.full(np.size(v), np.nan)
        OD_CO2_singleLayer = xco2_0*((1e-4)*NA*Abs_CO2/((Mair + MH2O*sh)*g))       #（1e-4）来源于cm到m的转换；NA来源于mol到molecule的转换
        OD_CO2_singleLayer_on = OD_CO2_singleLayer[:, 0]
        OD_CO2_singleLayer_off = OD_CO2_singleLayer[:, 1]
        OD_CO2[v==working_wavelength] = np.trapz(OD_CO2_singleLayer_on, Pre)
        OD_CO2[v==reference_wavelength] = np.trapz(OD_CO2_singleLayer_off, Pre)
        
        Pre_X = Pre.reshape(-1, 1)
        Nair = (1e-6)*Pre_X/(R*tem)                                                # mol/cm^3  
        Nh2o = sh*Nair                                                             # 这里用的是比湿，不是相对湿度
        OD_H2O = np.full(np.size(v), 0)
        OD_H2O_singleLayer = 100*NA*Nh2o*Abs_H2O                                   # 100 来源于m到cm的转换；NA来源于mol到molecule的转换                     
        OD_H2O_on_singleLayer = OD_H2O_singleLayer[:, 0]
        OD_H2O_off_singleLayer = OD_H2O_singleLayer[:, 1]
        Height_X = height.reshape(-1, )
        OD_H2O[v==working_wavelength] = np.trapz(OD_H2O_on_singleLayer, Height_X)
        OD_H2O[v==reference_wavelength] = np.trapz(OD_H2O_off_singleLayer, Height_X)
        
        OD_CO2_on = OD_CO2[v == working_wavelength]
        OD_CO2_off = OD_CO2[v == reference_wavelength]
        OD_H2O_on = OD_H2O[v == working_wavelength]
        OD_H2O_off = OD_H2O[v == reference_wavelength]
    
        Pon = (energy_on/Delt_teff)*optical_efficient*(A/pow(satellite_altitude,2))*O*ref*math.exp((-2)*(OD_CO2_on + OD_H2O_on + od))
        Poff = (energy_off/Delt_teff)*optical_efficient*(A/pow(satellite_altitude,2))*O*ref*math.exp((-2)*(OD_CO2_off + OD_H2O_off + od)) 
        
        Pback = pow(FOV, 2)*pow(A,2)*R_solar*ref/4
        # CNR_ON = Pon*intergal_G_factor*responsivity/(math.sqrt(filter_bandwidth*(2*e*pow(intergal_G_factor, 2)*excess_noise_factor*responsivity*(Pon + Pback) + pow(I_dark, 2) + 4*Kb*T/RL)))
        # CNR_OFF = Poff*intergal_G_factor*responsivity/(math.sqrt(filter_bandwidth*(2*e*pow(intergal_G_factor, 2)*excess_noise_factor*responsivity*(Poff + Pback) + pow(I_dark, 2) +4*Kb*T/RL)))
        
        Itensity_On = Pon*intergal_G_factor*responsivity
        Itensity_Off = Poff*intergal_G_factor*responsivity
        Noise_shots_On = filter_bandwidth*2*e*pow(intergal_G_factor, 2)*excess_noise_factor*responsivity*(Pon + Pback)
        Noise_shots_Off = filter_bandwidth*2*e*pow(intergal_G_factor, 2)*excess_noise_factor*responsivity*(Poff + Pback)
        Noise_dark = filter_bandwidth*pow(I_dark, 2)
        Noise_Tem = filter_bandwidth*4*Kb*T/RL
        
        Noise_On = math.sqrt(Noise_shots_On + Noise_dark + Noise_Tem)
        Noise_Off = math.sqrt(Noise_shots_Off + Noise_dark + Noise_Tem)
        
        Iensity_Pon = np.random.normal(Itensity_On, 0.1*Noise_On/Nshots)           # 噪声可能加太大了
        Iensity_Poff = np.random.normal(Itensity_Off, 0.1*Noise_Off/Nshots)
        
        P_ON[i] = Iensity_Pon
        P_OFF[i] = Iensity_Poff
        if Iensity_Poff/Iensity_Pon > 1:                                           # 保证DAOD有物理意义
            DAOD_CO2_IPDA[i] = 0.5*np.log(Iensity_Poff/Iensity_Pon)
        else:
            DAOD_CO2_IPDA[i] = np.nan
        if Iensity_Poff/Iensity_Pon > 1:                                           # 保证DAOD有物理意义
            DAOD_CO2_IPDA[i] = 0.5*np.log(Iensity_Poff/Iensity_Pon)
        else:
            DAOD_CO2_IPDA[i] = np.nan
            
        if abs(i/len(Data_CAL["Lat_CAL"])-0.3)<1e-3 and flag[0]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成30%\n') 
            #  print('正演完成30%')
             # sys.stdout.flush()
            flag[0]=False
        if abs(i/len(Data_CAL["Lat_CAL"])-0.6)<1e-3 and flag[1]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成60%\n') 
             # print('正演完成60%')
            #  sys.stdout.flush()
            flag[1]=False
        if abs(i/len(Data_CAL["Lat_CAL"])-0.9)<1e-3 and flag[2]:
            # with open(outputpath+'simulation.log','a') as logfile:
                # logfile.write('正演完成90%\n') 
            #  print('正演完成90%')
             # sys.stdout.flush()
            flag[2]=False

    Abs_CO2_ON_ALL = np.transpose(Abs_CO2_ON_ALL)
    Abs_CO2_OFF_ALL = np.transpose(Abs_CO2_OFF_ALL)
    Abs_H2O_ON_ALL = np.transpose(Abs_H2O_ON_ALL)
    Abs_H2O_OFF_ALL = np.transpose(Abs_H2O_OFF_ALL)


    # end = time.time()
    # runtime = end - start
    # print('正演模拟仿真运行时间：', runtime)
    # sys.stdout.flush()
    with open(outputpath+'result_daod.txt','w+') as logfile:
        logfile.write('DAOD最大值：'+str(round(np.nanmax(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD最小值：'+str(round(np.nanmin(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD平均值：'+str(round(np.nanmean(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD标准差：'+str(round(np.nanstd(DAOD_CO2_IPDA),3))+'\n'
        ) 
    with open(outputpath+'result_pon.txt', 'w+') as logfile:
        logfile.write('Pon最大值：{:.3e} W\n'.format(np.nanmax(P_ON)) +
                    'Pon最小值：{:.3e} W\n'.format(np.nanmin(P_ON)) +
                    'Pon平均值：{:.3e} W\n'.format(np.nanmean(P_ON)) +
                    'Pon标准差：{:.3e} W\n'.format(np.nanstd(P_ON))
        ) 
    with open(outputpath+'result_poff.txt', 'w+') as logfile:
        logfile.write('Poff最大值：{:.3e} W\n'.format(np.nanmax(P_OFF)) +
                    'Poff最小值：{:.3e} W\n'.format(np.nanmin(P_OFF)) +
                    'Poff平均值：{:.3e} W\n'.format(np.nanmean(P_OFF)) +
                    'Poff标准差：{:.3e} W\n'.format(np.nanstd(P_OFF))
        ) 
    
    print('DAOD最大值：'+str(round(np.nanmax(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD最小值：'+str(round(np.nanmin(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD平均值：'+str(round(np.nanmean(DAOD_CO2_IPDA),3))+'\n'
                       'DAOD标准差：'+str(round(np.nanstd(DAOD_CO2_IPDA),3))+'\n'
                       'Pon最大值：{:.3e} W\n'.format(np.nanmax(P_ON)) +
                       'Pon最小值：{:.3e} W\n'.format(np.nanmin(P_ON)) +
                       'Pon平均值：{:.3e} W\n'.format(np.nanmean(P_ON)) +
                       'Pon标准差：{:.3e} W\n'.format(np.nanstd(P_ON))+
                        'Poff最大值：{:.3e} W\n'.format(np.nanmax(P_OFF)) +
                        'Poff最小值：{:.3e} W\n'.format(np.nanmin(P_OFF)) +
                        'Poff平均值：{:.3e} W\n'.format(np.nanmean(P_OFF)) +
                        'Poff标准差：{:.3e} W\n'.format(np.nanstd(P_OFF)))
    sys.stdout.flush()
    
    #%% 正演结果保存
    # Abs_CO2_ON_ALL = np.transpose(Abs_CO2_ON_ALL)
    # Abs_CO2_OFF_ALL = np.transpose(Abs_CO2_OFF_ALL)
    # Abs_H2O_ON_ALL = np.transpose(Abs_H2O_ON_ALL)
    # Abs_H2O_OFF_ALL = np.transpose(Abs_H2O_OFF_ALL)
    Main_Results_Forward_Simulation_dict = {"Lat_CAL": Data_CAL["Lat_CAL"].tolist(), 
                                "Lon_CAL": Data_CAL["Lon_CAL"].tolist(),
                                "P_ON": P_ON.reshape(-1,),
                                "P_OFF": P_OFF.reshape(-1,),
                                "DAOD": DAOD_CO2_IPDA.reshape(-1,)}

    Main_Results_Forward_Simulation = pd.DataFrame(Main_Results_Forward_Simulation_dict)
    Main_Results_Forward_Simulation.to_csv(outputpath+"Main_Results_Forward_Simulation.csv")
    # Main_Results_Forward_Simulation = xr.Dataset({"P_ON": (("Lat_CAL"), P_ON.reshape(-1,)), 
    #                                             "P_OFF": (("Lat_CAL"), P_OFF.reshape(-1,))}, 
    #                                             coords={"Lat_CAL": Data_CAL["Lat_CAL"].values,
    #                                                     "Lon_CAL": Data_CAL["Lon_CAL"].values})
    # Main_Results_Forward_Simulation.to_netcdf(outputpath+"Main_Results_Forward_Simulation.nc")

    Auxiliary_Results_Forward_Simulation = xr.Dataset({"Height_CAL": (("Lat_CAL","Pressure"), Height_CAL), 
                                            "Specific_Humidity_CAL": (("Lat_CAL","Pressure"), SH_CAL), 
                                            "Temperature_CAL": (("Lat_CAL","Pressure"), Tem_CAL),
                                            "Abs_CO2_ON_ALL": (("Lat_CAL","Pressure"), Abs_CO2_ON_ALL), 
                                            "Abs_CO2_OFF_ALL": (("Lat_CAL","Pressure"), Abs_CO2_OFF_ALL), 
                                            "Abs_H2O_ON_ALL": (("Lat_CAL","Pressure"), Abs_H2O_ON_ALL), 
                                            "Abs_H2O_OFF_ALL": (("Lat_CAL","Pressure"), Abs_H2O_OFF_ALL)}, 
                                        coords={"Lat_CAL": Data_CAL["Lat_CAL"].values, 
                                                "Lon_CAL": Data_CAL["Lon_CAL"].values,
                                                "Pressure": Pre})
    Auxiliary_Results_Forward_Simulation.to_netcdf(outputpath+"Auxiliary_Results_Forward_Simulation.nc")
    #%% 画图
    pplt.rc.update(metacolor='white', fontsize=15, gridcolor='white',
                   figurefacecolor='black',axesfacecolor='black')
    fig = pplt.figure(share=False, refwidth=5, refaspect=(340, 104))
    ax = fig.subplot()
    ax.scatter(DAOD_CO2_IPDA,marker='.',color='yellow4')
    tick_values=np.arange(1,len(DAOD_CO2_IPDA),249)
    tick_CAL = Data_CAL["Lat_CAL"].values[::249]
    tick_labels = [f"{val:.0f}$^\circ$" for val in tick_CAL]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    file_name = outputpath+'plot_DAOD.png'
    ax.format(
        xlabel='Latitude',ylabel='DAOD', suptitle='Forward Simulation DAOD',
        suptitlecolor='white', gridcolor='white'
    )
    # fig.savefig(file_name,facecolor='black')
    fig.savefig(file_name,transparent=True)

    fig2 = pplt.figure(share=False, refwidth=5, refaspect=(340, 104))
    ax = fig2.subplot()
    ax.plot(P_ON,color='blue3',linewidth=1,label='$P_{ON}$',legend='ur')
    ax.plot(P_OFF,color='red5',linewidth=1,label='$P_{OFF}$',legend='ur')
    tick_values=np.arange(1,len(P_ON),249)
    tick_CAL = Data_CAL["Lat_CAL"].values[::249]
    tick_labels = [f"{val:.0f}$^\circ$" for val in tick_CAL]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    file_name = outputpath+'plot_P.png'
    ax.format(
        xlabel='Latitude',ylabel='Power', suptitle='IPDA Signal, unit:W',
        suptitlecolor='white', gridcolor='white'
    )

    fig2.savefig(file_name,transparent=True)
    # fig2.savefig(file_name,facecolor='black')
    pplt.rc.reset()
    # with open(outputpath+'simulation.log','a') as logfile:
    #     logfile.write('正演结果已保存\n') 
    #  print('正演结果已保存')
     # sys.stdout.flush()
    #  print("如您对程序结果有任何问题,请联系")
    #  sys.stdout.flush()
     # print("浙江大学刘东教授（liudongopt@zju.edu.cn）团队")
     # sys.stdout.flush()

def retrieval(Path):
    # config data reading
    config = configparser.ConfigParser()                                           # 启动ConfigParse
    config.read(Path+'/Config_Parameters.ini')                                     # 使用read()读取文件
    inputpath =Path+'/output_simulation/'
    outputpath=Path+'/output_retrieval/'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    # working_wavelength =config['NEWSETTINGS'].getfloat('working_wavelength')       
    # reference_wavelength= config['NEWSETTINGS'].getfloat('reference_wavelength')
    energy_on= config['NEWSETTINGS'].getfloat('energy_on')*(1e-3)
    energy_off= config['NEWSETTINGS'].getfloat('energy_off')*(1e-3)
    # logfile clear
    # logfile= open(outputpath+'retrieval.log','w+')
    # logfile.write('')
    # logfile.close()

    Mair = 28.964425278793996*(1e-3)                                               # 空气平均摩尔质量，单位kg/mol
    MH2O = 18*(1e-3)                                                               # 水的平均摩尔质量，单位kg/mol
    NA = 6.02214129*(1e+23)                                                        # 阿伏伽德罗常数 1/mol
    g = 9.8066                                                                     # 重力加速度
    R = 8.31441                                                                    # 普适气体常量J/(mol·k)

    # 数据读取
    ds1 = pd.read_csv(inputpath+"Main_Results_Forward_Simulation.csv")
    ds2 = xr.open_dataset(inputpath+"Auxiliary_Results_Forward_Simulation.nc")
    P_ON = ds1["P_ON"].values
    P_OFF = ds1["P_OFF"].values
    Height_CAL = ds2["Height_CAL"].values
    SH_CAL = ds2["Specific_Humidity_CAL"].values
    Tem_CAL = ds2["Temperature_CAL"].values
    Abs_CO2_ON_ALL = ds2["Abs_CO2_ON_ALL"].values
    Abs_CO2_OFF_ALL = ds2["Abs_CO2_OFF_ALL"].values
    Abs_H2O_ON_ALL = ds2["Abs_H2O_ON_ALL"].values
    Abs_H2O_OFF_ALL = ds2["Abs_H2O_OFF_ALL"].values
    Lat_CAL = ds2["Lat_CAL"].values
    Lon_CAL = ds2["Lon_CAL"].values
    Pre = ds2["Pressure"].values
    # ds1.close()
    ds2.close()
    # with open(outputpath+'retrieval.log','a') as logfile:
    #     logfile.write('数据读取已完成\n')
    #print('数据读取已完成')
    sys.stdout.flush()
    # 反演算法
    flag= [True,True,True]
    # v = np.array([working_wavelength, reference_wavelength])
    XCO2_retr = np.full(len(Lat_CAL), np.nan)
    for i in np.arange(len(Lat_CAL)):
        pon = P_ON[i]
        poff = P_OFF[i]
        height = Height_CAL[i,:]
        sh = SH_CAL[i,:]                                                           # python中SH_CAL[i,:]与SH_CAL[i]一样
        tem = Tem_CAL[i,:]
        abs_CO2_on = Abs_CO2_ON_ALL[i,:]
        abs_CO2_off = Abs_CO2_OFF_ALL[i,:]
        abs_H2O_on = Abs_H2O_ON_ALL[i,:]
        abs_H2O_off = Abs_H2O_OFF_ALL[i,:]
        
        Nair = (1e-6)*Pre/(R*tem) 
        Nh2o = sh*Nair
        od_H2O_on_singleLayer = 100*NA*Nh2o*abs_H2O_on
        od_H2O_off_singleLayer = 100*NA*Nh2o*abs_H2O_off
        od_H2O_on = np.trapz(od_H2O_on_singleLayer, height)
        od_H2O_off = np.trapz(od_H2O_off_singleLayer, height)
        DAOD_H2O = od_H2O_on - od_H2O_off
        
        if (poff*energy_on)/(pon*energy_off) > 1:
            DAOD = 0.5*np.log((poff*energy_on)/(pon*energy_off)) - DAOD_H2O
        else:
            DAOD = np.nan
        
        WF = NA*(1e-4)*(abs_CO2_on - abs_CO2_off)/((Mair + MH2O*sh)*g)
        IWF = np.trapz(WF, Pre)
        XCO2_retr[i] = DAOD/((1e-6)*IWF)
        
        if abs(i/np.size(Lat_CAL)-0.3)<1e-3 and flag[0]:
            # with open(outputpath+'retrieval.log','a') as logfile:
            #     logfile.write('反演完成30%\n')
            # sys.stdout.flush()
            flag[0]=False
        if abs(i/np.size(Lat_CAL)-0.6)<1e-3 and flag[1]:
            # with open(outputpath+'retrieval.log','a') as logfile:
            #     logfile.write('反演完成60%\n') 
            # print('反演完成60%')
            #  sys.stdout.flush()
            flag[1]=False
        if abs(i/np.size(Lat_CAL)-0.9)<1e-3 and flag[2]:
            # with open(outputpath+'retrieval.log','a') as logfile:
            #     logfile.write('反演完成90%\n')
             # print('反演完成90%')
             # sys.stdout.flush()
            flag[2]=False

    #%% 反演结果保存
    Main_Results_Retrival_Simulation_dict = {"Lat_CAL": Lat_CAL, 
                                             "Lon_CAL": Lon_CAL,
                                            "XCO2_retrival": XCO2_retr}
    Main_Results_Retrival_Simulation = pd.DataFrame(Main_Results_Retrival_Simulation_dict)
    Main_Results_Retrival_Simulation.to_csv(outputpath+"Main_Results_Retrival_Simulation.csv", index=False)
    
    with open(outputpath+'result_r.txt','w+') as logfile:
        logfile.write('XCO2最大值：'+str(round(np.nanmax(XCO2_retr),3))+'ppm\n'
                       'XCO2最小值：'+str(round(np.nanmin(XCO2_retr),3))+'ppm\n'
                       'XCO2平均值：'+str(round(np.nanmean(XCO2_retr),3))+'ppm\n'
                       'XCO2标准差：'+str(round(np.nanstd(XCO2_retr),3))+'ppm\n'
        ) 
    

    print('XCO2最大值：'+str(round(np.nanmax(XCO2_retr),3))+'ppm\n'
                       'XCO2最小值：'+str(round(np.nanmin(XCO2_retr),3))+'ppm\n'
                       'XCO2平均值：'+str(round(np.nanmean(XCO2_retr),3))+'ppm\n'
                       'XCO2标准差：'+str(round(np.nanstd(XCO2_retr),3))+'ppm\n')
    sys.stdout.flush()
    
    
    #%% 画图
    warnings.filterwarnings('ignore')
    XCO2_retr_fig = XCO2_retr
    pplt.rc.update(metacolor='white', fontsize=15, gridcolor='white',
                    figurefacecolor='black',axesfacecolor='black')
    fig = pplt.figure(share=False, refwidth=5, refaspect=(2, 1))
    pplt.rc.reso = 'med'
    ax = fig.subplot(proj='cyl')
    ax.format(
        labels=True,
        # land=True,landcolor='white',landzorder=0,
        coast=True, coastcolor='white',coastzorder=0,
        gridminor=True, lonlabels='b', latlabels='l',
        suptitle='Retrival Simulation XCO2, unit:ppm',
        suptitlecolor='white',labelsize=15
    )
    ax.scatter(
        Lon_CAL,Lat_CAL, c=XCO2_retr_fig[::1], zorder=1,
        s = 20, marker='o', robust=True, extend='both',
        discrete=True, levels=5,
        colorbar='r', cmap='YlOrRd', vmin=395, vmax=405,
    )

    file_name = outputpath+'plot_r.png'
    fig.savefig(file_name, transparent=True)
    # fig.savefig(file_name, facecolor='black')
    pplt.rc.reset()
    # with open(outputpath+'retrieval.log','a') as logfile:
    #     logfile.write('反演结果已保存\n')
     # print('反演结果已保存')
     # sys.stdout.flush()
    #  print("如您对程序结果有任何问题,请联系")
    #  sys.stdout.flush()
    #  print("浙江大学刘东教授（liudongopt@zju.edu.cn）团队")
    #  sys.stdout.flush()