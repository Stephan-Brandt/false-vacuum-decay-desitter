# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:28:53 2021

@author: Juan
"""

import os, glob
import pandas as pd

#path = "d:\\Juan\\Documents\\dS Project Data\\dataGreens"
myPath = "d:\\Juan\\Documents\\dS Project Data\\dataGreensFV\\"
os.chdir(myPath)

outPath = os.path.join(myPath,"concatenated\\")
print(outPath)

for i in range(2,101):
    ell = str(i)
    fileNamePattern = r"greenSparamFV-"+ ell +"[-|.]*csv"
    rawList = glob.glob(fileNamePattern)
    all_files = [rawList[2],rawList[0],rawList[1]]

    df_from_each_file = (pd.read_csv(f, sep=',',header=None) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=False, names=None)
    fileName = "gParamsFV"+ ell +"merged.csv"
    df_merged.to_csv( os.path.join(outPath,fileName), index=False,header=None )
    print("File: "+ fileName + " has been save succesfully.")
    
print("Merging raw data finished succesfully.")
