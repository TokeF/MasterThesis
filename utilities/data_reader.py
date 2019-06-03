import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(fileName : str):
    #read .xyz datafile into pandas data frame
    dataFrame = pd.read_csv(fileName, sep='\s+',header=[18])
    assert dataFrame.columns[1] == 'TIMESTAMP', "Header is wrong. Location 1 yield: " + str(dataFrame.columns[1])
    #correct header names
    #first column is a \ due to data file. Thus labels are shifted one back. Column 179 is empty and thus deleted
    header = dataFrame.columns.drop('/')
    dataFrame.drop("DBDT_INUSE_Ch2GT33", axis=1, inplace=True)
    dataFrame.columns = header
    #assert that there are no NAN
    assert not dataFrame.isnull().values.any(), "NAN value in data: " + str(dataFrame.isnull().sum().sum())
    return dataFrame

def load_data2(fileName : str, gF : int, gT : int):
    dataFrame = load_data(fileName)
    gtimes = pd.read_csv(fileName, sep='\s+', skiprows=17, nrows=1, header=None)
    gtimes = gtimes.iloc[:, gF:gT+1].values
    gFrom = 'DBDT_Ch2GT' + str(gF)
    gTo = 'DBDT_Ch2GT' + str(gT)
    lblFrom = 'DBDT_INUSE_Ch2GT' + str(gF)
    lblTo = 'DBDT_INUSE_Ch2GT' + str(gT)
    dbdt = dataFrame.loc[:,gFrom:gTo].values
    lbl = dataFrame.loc[:,lblFrom:lblTo].values
    dumdbdt = dbdt == 99999
    dumlbl = lbl == 99999
    assert not dumdbdt.any().any(), 'Dummy values in DBDT present: ' + str(dumdbdt.any().sum().sum())
    assert not dumlbl.any().any(), 'Dummy values in label present: ' + str(dumlbl.any().sum().sum())
    assert (lbl.T == lbl[:, 0]).any(), 'Labels are not equal'
    lbl = lbl[:, 0]
    return dataFrame, dbdt, lbl, dataFrame.loc[:, 'TIMESTAMP'].values, gtimes

#remove soundings around edges
#notice if first hole is before cutoff edge we have a problem.. and last
def remove_edge(timestamp, dbdt, lbl, nremove):
    timestampOG = timestamp
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    timestamp = np.insert(timestamp, 0, timestamp[0])
    timestamp = np.reshape(timestamp, (timestamp.shape[0], 1))
    t_diff = np.diff(timestamp, axis=0)
    t_diff = mm.fit_transform(t_diff)
    # sns.distplot(t_diff, 1000)
    t_diff = np.where(t_diff > 0.011) #0.011)
    for i, idx in enumerate(t_diff[0]):
        b = nremove
        dbdt[idx-b : idx + b, :] = np.nan
        lbl[idx-b : idx + b] = np.nan
        timestampOG[idx-b : idx + b] = np.nan
    #super weird, I dunno why i cant just do as with the array...
    lbl = lbl[lbl >= 0]
    timestampOG = timestampOG[timestampOG >= 0]
    return dbdt[~np.isnan(dbdt).any(axis=1)], lbl, timestampOG


def timestampToTime(timestamp):
    b = timestamp - timestamp[0]
    customdate = datetime.datetime(2016, 1, 1, 13, 30)
    return [customdate + datetime.timedelta(days=t) for t in b]
