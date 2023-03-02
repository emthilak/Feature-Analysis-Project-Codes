# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:46:02 2023

@author: emt38
"""

#Testing nd2reader functionality over large set of images
import nd2reader
import numpy as np
import matplotlib.pyplot as plt
import os

RG_Data =[r'G:\.shortcut-targets-by-id\1-15-JizXAXpNxTulSmAe9cmPge3bxBld\ULK Drugs Data\07142022_ULK101_SBI_Rap\Experimental Data\posttreatment.nd2', #ULK SBI Data
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/07142022_ULK101_SBI_Rap/Experimental Data/pretreatment (1).nd2',
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/07142022_ULK101_SBI_Rap/Experimental Data/pretreatmentpart2.nd2',
          r'G:\.shortcut-targets-by-id\1-15-JizXAXpNxTulSmAe9cmPge3bxBld\ULK Drugs Data\08062022-MRT68921-ULK101\ExperimentalData/minus20.nd2', #MRT68921 Data
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08062022-MRT68921-ULK101/ExperimentalData/posttreat.nd2', 
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08062022-MRT68921-ULK101/ExperimentalData/pretreatment.nd2',
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08192022-ULK101-SBI0206965/ExperimentalData/minus20.nd2', #SBI0206965 Data
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08192022-ULK101-SBI0206965/ExperimentalData/posttreatment.nd2', 
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08192022-ULK101-SBI0206965/ExperimentalData/prettreatment.nd2', 
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08272022_MRT307_SBP7455/Experimental Data/minus20.nd2', #MRT307_SBP7455 Data
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08272022_MRT307_SBP7455/Experimental Data/post.nd2', 
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/08272022_MRT307_SBP7455/Experimental Data/pretreatment.nd2',
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/9102022-MRT-67307-SBP-7455/Experimental Data/minus20.nd2', #MRT-67307-SBP-7455 Data
          r'G:\.shortcut-targets-by-id\1-15-JizXAXpNxTulSmAe9cmPge3bxBld\ULK Drugs Data\9102022-MRT-67307-SBP-7455\Experimental Data/posttreatment.nd2',
          r'G:/.shortcut-targets-by-id/1-15-JizXAXpNxTulSmAe9cmPge3bxBld/ULK Drugs Data/9102022-MRT-67307-SBP-7455/Experimental Data/pretreatment.nd2']

VG_Data = [r'D:/Eshan/motherload of data/pretreatment (1).nd2',
           r'D:\Eshan\motherload of data/pretreatmentpart2 (1).nd2']

NBA_Data = [r'G:\.shortcut-targets-by-id\19h0G23OJAY4hBZFj9u6nPDSgFPcagpah\Tiff_files\04222022_huh7_zikainfection/04222022_huh7_zikainfection_40hrs.nd2',
            r'G:\.shortcut-targets-by-id\19h0G23OJAY4hBZFj9u6nPDSgFPcagpah\Tiff_files\09232022_huh7_ZIKV_organelles/09232022_huh7_ZIKV_organelles_36hrs001.nd2',
            r'G:\.shortcut-targets-by-id\19h0G23OJAY4hBZFj9u6nPDSgFPcagpah\Tiff_files\09232022_huh7_ZIKV_organelles/09232022_huh7_ZIKV_organelles_36hrs002.nd2',
            r'G:\.shortcut-targets-by-id\19h0G23OJAY4hBZFj9u6nPDSgFPcagpah\Tiff_files\11102022_huh7_ZIKV_organelles_2_test/09232022_huh7_ZIKV_organelles2_36hrs.nd2',
            r'G:\.shortcut-targets-by-id\19h0G23OJAY4hBZFj9u6nPDSgFPcagpah\Tiff_files\11292022_Huh7_HeLa_ER_Tracking_Test/09232022_huh7_ZIKV_organelles2_36hrs002.nd2']

NSB_Data = [r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB02_ChannelTRITC_NSB,GFPNSB_Seq0000.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB03_ChannelTRITC_NSB,GFPNSB_Seq0006.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB04_ChannelTRITC_NSB,GFPNSB_Seq0012.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB05_ChannelTRITC_NSB,GFPNSB_Seq0018.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB06_ChannelTRITC_NSB,GFPNSB_Seq0024.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/01_pretreatment_1hours__12162020_17755/20201216_102037_157/WellB07_ChannelTRITC_NSB,GFPNSB_Seq0030.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/05152022_B11_starvation_Recovery/05152022_post.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/05152022_B11_starvation_Recovery/05152022_post001_addback.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/05152022_B11_starvation_Recovery/05152022_pretreatment.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/05152022_B11_starvation_Recovery/05152022_pretreatment001.nd2',
            r'G:/.shortcut-targets-by-id/1-tGVzVMrTtlWoQod60ygHGkSQ0kPcmvJ/rawdata/05152022_B11_starvation_Recovery/05152022_pretreatment002.nd2']



totalfiles = len(RG_Data) + len(VG_Data) + len(NBA_Data) + len(NSB_Data)
am = 'v' #axis missing

#Testing RG Data
RG_errs = []
for file_idx in range(len(RG_Data)):
    test = nd2reader.ND2Reader(RG_Data[file_idx])
    if am not in test.axes:
        RG_errs.append(file_idx)
        
#Testing VG Data
VG_errs = []
for file_idx in range(len(VG_Data)):
    test = nd2reader.ND2Reader(VG_Data[file_idx])
    if am not in test.axes:
        VG_errs.append(file_idx)
        
#Testing NBA Data
NBA_errs = []
for file_idx in range(len(NBA_Data)):
    test = nd2reader.ND2Reader(NBA_Data[file_idx])
    if am not in test.axes:
        NBA_errs.append(file_idx)

#Testing NSB Data
NSB_errs = []
for file_idx in range(len(NSB_Data)):
    test = nd2reader.ND2Reader(NSB_Data[file_idx])
    if am not in test.axes:
        NSB_errs.append(file_idx)
        
        
if (len(RG_errs) + len(VG_errs) + len(NBA_errs) + len(NSB_errs)> 0):
    print('Errors Found!')
else:
    print('No Errors')