#!/usr/bin/env python
# coding: utf-8

# Calculate the following equity characteristics, and the **formulas** are refering the US equity python code.
# - Market Equity
# - Book-to-Market Ratio
# 
# this demo is on firms:
# - headquartered (LOC) in CHN
# - publicly listed (EXCHG) in China SH or SZ
# - currency (CURCDD) is CNY
# - registration country (FIC) can be outside China, e.g. ALIBABA in Cayman Island
# 
# exchg:
# - cn sh 249
# - cn sz 250

# In[320]:


import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[321]:


# ###################
# # Connect to WRDS #
# ###################
# conn = wrds.Connection()


# # Download data and store locally
# 
# Downloading takes a long time, but if you save the data in .pkl locally, it would be very fast.

# ## fundq by country (headquarter)

# In[322]:


# fundq = conn.raw_sql("""
#                       select *
#                       from comp.g_fundq
#                       where datadate > '01/01/2015'
#                       and loc = 'HKG'
#                       """)

# fundq = fundq.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
#  with open('./fundq_hkg_2015.pkl', 'wb') as f:
#    pkl.dump(fundq, f)


# In[323]:


# fundq = conn.raw_sql("""
#                       select *
#                       from comp.g_fundq
#                       where datadate > '01/01/2000'
#                       and loc = 'HKG'
#                       """)

#fundq = fundq.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./fundq_hkg_2000.pkl', 'wb') as f:
#    pkl.dump(fundq, f)


# ## secd by country (headquarter)

# In[324]:


# secd = conn.raw_sql("""
#                       select *
#                       from comp.g_secd
#                       where datadate > '01/01/2015'
#                       and loc = 'HKG'
#                       """)
#secd = secd.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./secd_hkg_2015.pkl', 'wb') as f:
#     pkl.dump(secd, f)


# In[325]:


# secd = conn.raw_sql("""
#                       select *
#                       from comp.g_secd
#                       where datadate > '01/01/2000'
#                       and loc = 'HKG'
#                       """)
#secd = secd.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./secd_hkg_2000.pkl', 'wb') as f:
#     pkl.dump(secd, f)


# # Read secd data

# In[326]:


secd = pd.read_pickle('./secd_hkg_2000.pkl')


# ### filters on secd

# In[327]:


secd = secd[~secd['isin'].isna()]   # international id
secd = secd[~secd['sedol'].isna()]   # international id
secd = secd[~secd['cshoc'].isna()]  # number of common shares
secd = secd[secd['tpci']=='0']      # issue tyoe code, 0 for equities 
secd = secd[ (secd['exchg'] == 170)] # Hongkong
# secd = secd[secd['curcdd']=='HKD']  # currency


# ### calculate me
# 
# - sort the different issues of the same firm
# - Be careful about cross-listing, like Alibaba in US and HK, SinoPetro in CN, HK, US.
# - check this later.

# In[328]:


secd['me'] = secd['prccd']*secd['cshoc']
secd = secd.sort_values(['gvkey','datadate','me','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
secd.index = range(len(secd.index))


# ### calculate daily returns

# In[329]:


secd['prc_adj'] = secd['prccd']/secd['ajexdi']
secd['prc_trfd'] = secd['prccd']/secd['ajexdi']*secd['trfd']
secd['prc_trfd_last_day'] = secd.groupby(['gvkey'])['prc_trfd'].shift(1)
secd['ret'] = secd['prc_trfd']/secd['prc_trfd_last_day']-1


# In[330]:


secd.columns


# In[331]:


varlist=['gvkey', 'exchg','tpci', 'prcstd', 'loc','fic', 'iid','sedol', 'isin','datadate', 'cshoc','conm','monthend','curcdd', 'prccd','prc_trfd','prc_trfd_last_day','ajexdi','trfd','prc_adj','me','ret']
secd = secd[varlist]


# returns

# In[332]:


secd['ret_plus_one'] = secd['ret']+1
secd['monthid'] = secd.groupby(['gvkey']).cumsum()['monthend']
secd['cumret'] = secd[['gvkey','monthid','ret_plus_one']].groupby(['gvkey']).cumprod()['ret_plus_one']


# # Work on monthly frequency

# In[333]:


secm = secd[secd['monthend']==1]
secm['cumret_last_month'] = secm.groupby('gvkey').shift(1)['cumret']
secm['retm'] = secm['cumret']/secm['cumret_last_month']-1


# In[334]:


tmp=secm.groupby(['gvkey','exchg']).nunique()['isin']
tmp[tmp>1]
# if you remove the currency or exchange filter, you will see one gvkey links to multiple iid/isin/sedol
# now, for the China case, we have one-one-mapping between gvkey and iid (issues)


# # Read fundq

# In[335]:


fundq = pd.read_pickle('./fundq_hkg_2000.pkl')


# In[336]:


tmp = fundq[['datadate','seqq']]
tmp['seqq_na'] = tmp['seqq'].isna()
na_num = tmp.groupby('datadate').sum()['seqq_na']
tot_num = tmp.groupby('datadate').count()['seqq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# ### filters on fundq

# In[337]:


# fundq = fundq[~fundq['isin'].isna()]   # international id
# fundq = fundq[~fundq['sedol'].isna()]   # international id
fundq = fundq[fundq['exchg'] == 170] # hongkong


# In[338]:


fundamental_varlist=[
    # id
    'gvkey', 'indfmt', 'consol', 'popsrc', 'datafmt','exchg', 'loc','fic', 'sedol', 'isin','datadate','pdateq','fdateq','fyr',
    # varaibles we want 
    'ibq','iby',
    'seqq','txdbq','txtq','pstkq','dpy','dpq','atq',
    'cheq','actq','gdwlq','intanq','ceqq',
    'ivaoq','dlcq','dlttq','mibq','saleq','saley',
    'ltq','ppentq','revtq','cogsq',
    'rectq','acoq','apq','lcoq','loq','invtq','aoq','xintq','xsgaq','oiadpq','oancfy'
    ]
fundq = fundq[fundamental_varlist]
fundq.head(50)


# In[339]:


# fundq = fundq[~fundq['pdateq'].isna()] # some empty observations in fundq, you can check this with the next commented code
# fundq[fundq['gvkey']=='029530'].head(15)


# In[340]:


fundq = fundq.sort_values(['gvkey','datadate','exchg','isin','sedol','seqq'])


# In[341]:


tmp=fundq.groupby(['gvkey','exchg']).nunique()['isin']
tmp[tmp>1]
# make sure, one gvkey-exchange has only one isin/sedol


# ### drop some observations which losses critical info

# In[342]:


# print(fundq.shape)
# fundq = fundq[~fundq['seqq'].isna()]
# fundq = fundq[~fundq['ibq'].isna()]
# print(fundq.shape)


# ### impute some variables to zero

# In[343]:


fundq['txdbq'] = fundq['txdbq'].fillna(0)
fundq['txtq'] = fundq['txtq'].fillna(0)
fundq['pstkq'] = fundq['pstkq'].fillna(0)

fundq['mibq'] = fundq['mibq'].fillna(0)
fundq['dlcq'] = fundq['dlcq'].fillna(0)
fundq['ivaoq'] = fundq['ivaoq'].fillna(0)
fundq['dlttq'] = fundq['dlttq'].fillna(0)


# ## Calculate fundamental information
# 
# this is for Lingqiao to continue

# In[344]:


fundq['beq'] = fundq['seqq'] + fundq['txdbq'] + fundq['txtq'] - fundq['pstkq']


# In[345]:


# other information


# # Merge fundq and secm

# In[346]:


fundq['datadate'] = pd.to_datetime(fundq['datadate'])
# join date is jdate
# quarterly fundamentals are expected to report later than the datadate
# 3 month is enough for the reporting process
# thus, we don't have forseeing-data problem
fundq['jdate'] = fundq['datadate'] + MonthEnd(3)
fundq = fundq.sort_values(['gvkey','datadate','exchg','isin','sedol'])


# In[347]:


tmp = fundq[['datadate','seqq']]
tmp['seqq_na'] = tmp['seqq'].isna()
na_num = tmp.groupby('datadate').sum()['seqq_na']
tot_num = tmp.groupby('datadate').count()['seqq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[348]:


secm['datadate'] = pd.to_datetime(secm['datadate'])
secm['jdate'] = secm['datadate'] + MonthEnd(0)
secm = secm.sort_values(['gvkey','datadate','exchg','isin','sedol'])
secm = secm[['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin', 'datadate', 
             'cshoc', 'prccd', 'me', 'retm', 'jdate']]


# In[349]:


fqsm = pd.merge(secm, fundq, how='left', on=['gvkey','jdate','exchg','isin','sedol','loc','fic'])


# In[350]:


tmp = fqsm[fqsm['gvkey']=='029530']
tmp.head(20)


# # Forward Fill the Fundq info to Empty Month

# In[351]:


fqsm.columns


# In[352]:


fqsm.columns = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin', 'datadate_secm',
       'cshoc', 'prccd', 'me', 'retm', 'jdate', 'indfmt', 'consol', 'popsrc',
       'datafmt', 'datadate_fundq', 'pdateq', 'fdateq', 'fyr', 'ibq', 'iby',
       'seqq', 'txdbq', 'txtq', 'pstkq', 'dpy', 'dpq','atq', 'cheq','actq','gdwlq','intanq','ceqq',
        'ivaoq','dlcq','dlttq','mibq','saleq','saley','ltq','ppentq','revtq','cogsq',
        'rectq','acoq','apq','lcoq','loq','invtq','aoq','xintq','xsgaq','oiadpq','oancfy','beq']


# In[353]:


fqsm = fqsm.sort_values(['gvkey','jdate','isin','sedol']) # order by gvkey, date, issue id, other id's


# In[354]:


fqsm['pdateq'] = fqsm.groupby('gvkey')['pdateq'].fillna(method='ffill')
fqsm['fdateq'] = fqsm.groupby('gvkey')['fdateq'].fillna(method='ffill')
fqsm['ibq'] = fqsm.groupby('gvkey')['ibq'].fillna(method='ffill')
fqsm['beq'] = fqsm.groupby('gvkey')['beq'].fillna(method='ffill')
#
# fqsm['iby'] = fqsm.groupby('gvkey')['iby'].fillna(method='ffill')


# In[355]:


fqsm['me'] = fqsm['me']/1e6 # 1e6 is one million


# In[356]:


fqsm['bm'] = fqsm['beq']/fqsm['me']
fqsm['mb'] = fqsm['me']/fqsm['beq']


# In[357]:


fqsm.columns.values


# In[358]:


tmp = fqsm[fqsm['gvkey']=='253759'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','beq','me','bm','mb']]
tmp.tail(20) # byd sz
# xinhe checked this, fundq misses byd sz exchg 250, but only covers byd hk exchg 170


# In[359]:


tmp = fqsm[fqsm['gvkey']=='251321'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','beq','me','bm','mb']]
tmp.tail(20) # maotai
# you can check the MB as 市净率 with any stock app, it is 17.03 at 2021-03-26, which is almost the same as our number


# In[360]:


fundq.columns


# In[361]:


secd.columns


# In[362]:


fqsm.columns


# In[363]:


def ttm4(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm4
    """
    lag = pd.DataFrame()
    for i in range(3,10,3):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('gvkey')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s3' % series] + lag['%s6' % series] + lag['%s9' % series]
    return result
# changes from accounting_60.py: shift(3), shift(6), shift(9)

def ttm12(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm12
    """
    lag = pd.DataFrame()
    for i in range(1, 12):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('gvkey')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series] +             lag['%s4' % series] + lag['%s5' % series] + lag['%s6' % series] + lag['%s7' % series] +             lag['%s8' % series] + lag['%s9' % series] + lag['%s10' % series] + lag['%s11' % series]
    return result


# In[364]:


fqsm['earnings'] = ttm4('ibq',fqsm)
# fqsm['earnings'] = fqsm['ibq'] + fqsm['ibq'].shift(3) + fqsm['ibq'].shift(6) + fqsm['ibq'].shift(9)
fqsm['ep'] = fqsm['earnings'] / fqsm['me']
fqsm['pe'] = fqsm['me'] / fqsm['earnings']

# fqsm['ep2'] = ttm4('ibq',fqsm) / fqsm['me'] * 1e6
# fqsm['pe2'] = fqsm['me']/ttm4('ibq',fqsm) / 1e6


# In[365]:


tmp = fqsm[fqsm['gvkey']=='251321'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','beq','me','bm','mb','ibq','ep','pe','ibq','iby','earnings']]
tmp.tail(20) # maotai


# In[366]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','beq','me','bm','mb','ibq','ep','pe','ibq','iby','earnings']]
tmp.tail(40) # midea


# In[367]:


#cp
#dpq fillna
fqsm['dpq'] = fqsm.groupby('gvkey')['dpq'].fillna(method='ffill')
fqsm['cf'] = ttm4('ibq',fqsm) + ttm4('dpq',fqsm)
fqsm['cp'] = fqsm['cf']/fqsm['me']


# In[368]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','beq','me','bm','mb','ibq','dpq','cf','cp']]
tmp.tail(20)


# In[369]:


# agr
fqsm['atq'] = fqsm.groupby('gvkey')['atq'].fillna(method='ffill')
fqsm['atq_l4'] = fqsm.groupby('gvkey')['atq'].shift(12)
fqsm['agr'] = (fqsm['atq'] - fqsm['atq_l4']) / fqsm['atq_l4']


# In[370]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','atq','agr']]
tmp.tail(20)


# In[371]:


# alm
fqsm['cheq'] = fqsm.groupby('gvkey')['cheq'].fillna(method='ffill')
fqsm['actq'] = fqsm.groupby('gvkey')['actq'].fillna(method='ffill')
fqsm['gdwlq'] = fqsm.groupby('gvkey')['gdwlq'].fillna(method='ffill')
fqsm['intanq'] = fqsm.groupby('gvkey')['intanq'].fillna(method='ffill')
fqsm['ceqq'] = fqsm.groupby('gvkey')['ceqq'].fillna(method='ffill')
fqsm['ala'] = fqsm['cheq'] + 0.75*(fqsm['actq']-fqsm['cheq'])+                 0.5*(fqsm['atq']-fqsm['actq']-fqsm['gdwlq']-fqsm['intanq'])
fqsm['alm'] = fqsm['ala']/(fqsm['atq']+fqsm['me']-fqsm['ceqq'])


# In[372]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','me','cheq','actq','gdwlq','intanq','ceqq','atq','ala','alm']]
tmp.tail(20)


# In[373]:


# ato
fqsm['ivaoq'] = fqsm.groupby('gvkey')['ivaoq'].fillna(method='ffill')
fqsm['dlcq'] = fqsm.groupby('gvkey')['dlcq'].fillna(method='ffill')
fqsm['dlttq'] = fqsm.groupby('gvkey')['dlttq'].fillna(method='ffill')
fqsm['mibq'] = fqsm.groupby('gvkey')['mibq'].fillna(method='ffill')
fqsm['pstkq'] = fqsm.groupby('gvkey')['pstkq'].fillna(method='ffill')
fqsm['saleq'] = fqsm.groupby('gvkey')['saleq'].fillna(method='ffill')
fqsm['noa'] = (fqsm['atq']-fqsm['cheq']-fqsm['ivaoq'])-                 (fqsm['atq']-fqsm['dlcq']-fqsm['dlttq']-fqsm['mibq']-fqsm['pstkq']-fqsm['ceqq'])/fqsm['atq_l4']
fqsm['noa_l4'] = fqsm.groupby(['gvkey'])['noa'].shift(12)
fqsm['ato'] = fqsm['saleq']/fqsm['noa_l4']


# In[374]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','atq','cheq','ivaoq','dlcq','dlttq','mibq','pstkq','ceqq','noa','noa_l4','saleq','ato']]
tmp.tail(20)


# In[375]:


# cash
fqsm['cash'] = fqsm['cheq']/fqsm['atq']


# In[376]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','atq','cheq','cash']]
tmp.tail(20)


# In[377]:


# cashdebt
fqsm['ltq'] = fqsm.groupby(['gvkey'])['ltq'].fillna(method='ffill')
fqsm['ltq_l4'] = fqsm.groupby(['gvkey'])['ltq'].shift(12)
fqsm['cashdebt'] = (ttm4('ibq', fqsm) + ttm4('dpq', fqsm))/((fqsm['ltq']+fqsm['ltq_l4'])/2)


# In[378]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','prccd','atq','cheq','ltq','ltq_l4','cash']]
tmp.tail(20)


# In[379]:


#chpm
fqsm['ibq4'] = ttm4('ibq', fqsm)
fqsm['saleq4'] = ttm4('saleq', fqsm)
fqsm['saleq4'] = np.where(fqsm['saleq4'].isnull(), fqsm['saley'], fqsm['saleq4'])
fqsm['ibq4_l1'] = fqsm.groupby(['gvkey'])['ibq4'].shift(3)
fqsm['saleq4_l1'] = fqsm.groupby(['gvkey'])['saleq4'].shift(3)
fqsm['chpm'] = (fqsm['ibq4']/fqsm['saleq4'])-(fqsm['ibq4_l1']/fqsm['saleq4_l1'])


# In[380]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','ibq','saleq','ibq4','saleq4','ibq4_l1','saleq4_l1','chpm']]
tmp.tail(20)


# In[381]:


#chtx
fqsm['txtq'] = fqsm.groupby('gvkey')['txtq'].fillna(method='ffill')
fqsm['txtq_l4'] = fqsm.groupby(['gvkey'])['txtq'].shift(12)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(12)
fqsm['chtx'] = (fqsm['txtq']-fqsm['txtq_l4'])/fqsm['atq_l4']


# In[382]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','txtq','txtq_l4','atq_l4','chtx']]
tmp.tail(20)


# In[383]:


#cinvest
fqsm['ppentq'] = fqsm.groupby('gvkey')['ppentq'].fillna(method='ffill')
fqsm['ppentq_l1'] = fqsm.groupby(['gvkey'])['ppentq'].shift(3)
fqsm['ppentq_l2'] = fqsm.groupby(['gvkey'])['ppentq'].shift(6)
fqsm['ppentq_l3'] = fqsm.groupby(['gvkey'])['ppentq'].shift(9)
fqsm['ppentq_l4'] = fqsm.groupby(['gvkey'])['ppentq'].shift(12)
fqsm['saleq_l1'] = fqsm.groupby(['gvkey'])['saleq'].shift(3)
fqsm['saleq_l2'] = fqsm.groupby(['gvkey'])['saleq'].shift(6)
fqsm['saleq_l3'] = fqsm.groupby(['gvkey'])['saleq'].shift(9)

fqsm['c_temp1'] = (fqsm['ppentq_l1'] - fqsm['ppentq_l2']) / fqsm['saleq_l1']
fqsm['c_temp2'] = (fqsm['ppentq_l2'] - fqsm['ppentq_l3']) / fqsm['saleq_l2']
fqsm['c_temp3'] = (fqsm['ppentq_l3'] - fqsm['ppentq_l4']) / fqsm['saleq_l3']

fqsm['c_temp1'] = (fqsm['ppentq_l1'] - fqsm['ppentq_l2']) / 0.01
fqsm['c_temp2'] = (fqsm['ppentq_l2'] - fqsm['ppentq_l3']) / 0.01
fqsm['c_temp3'] = (fqsm['ppentq_l3'] - fqsm['ppentq_l4']) / 0.01

fqsm['cinvest'] = ((fqsm['ppentq'] - fqsm['ppentq_l1']) / fqsm['saleq'])                       -(fqsm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))
fqsm['cinvest'] = np.where(fqsm['saleq']<=0, ((fqsm['ppentq'] - fqsm['ppentq_l1']) / 0.01)
                                -(fqsm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), fqsm['cinvest'])

fqsm = fqsm.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)


# In[384]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','ppentq','cinvest']]
tmp.tail(20)


# In[385]:


#depr
fqsm['depr'] = ttm4('dpq', fqsm)/fqsm['ppentq']


# In[386]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','dpq','depr']]
tmp.tail(20)


# In[387]:


#gma
fqsm['revtq'] = fqsm.groupby('gvkey')['revtq'].fillna(method='ffill')
fqsm['cogsq'] = fqsm.groupby('gvkey')['cogsq'].fillna(method='ffill')
fqsm['revtq4'] = ttm4('revtq', fqsm)
fqsm['cogsq4'] = ttm4('cogsq', fqsm)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(12)
fqsm['gma'] = (fqsm['revtq4']-fqsm['cogsq4'])/fqsm['atq_l4']


# In[388]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','gma']]
tmp.tail(20)


# In[389]:


#grltnoa
fqsm['rectq'] = fqsm.groupby('gvkey')['rectq'].fillna(method='ffill')
fqsm['acoq'] = fqsm.groupby('gvkey')['acoq'].fillna(method='ffill')
fqsm['apq'] = fqsm.groupby('gvkey')['apq'].fillna(method='ffill')
fqsm['lcoq'] = fqsm.groupby('gvkey')['lcoq'].fillna(method='ffill')
fqsm['loq'] = fqsm.groupby('gvkey')['loq'].fillna(method='ffill')
fqsm['invtq'] = fqsm.groupby('gvkey')['invtq'].fillna(method='ffill')
fqsm['aoq'] = fqsm.groupby('gvkey')['aoq'].fillna(method='ffill')

fqsm['rectq_l4'] = fqsm.groupby(['gvkey'])['rectq'].shift(12)
fqsm['acoq_l4'] = fqsm.groupby(['gvkey'])['acoq'].shift(12)
fqsm['apq_l4'] = fqsm.groupby(['gvkey'])['apq'].shift(12)
fqsm['lcoq_l4'] = fqsm.groupby(['gvkey'])['lcoq'].shift(12)
fqsm['loq_l4'] = fqsm.groupby(['gvkey'])['loq'].shift(12)
fqsm['invtq_l4'] = fqsm.groupby(['gvkey'])['invtq'].shift(12)
fqsm['ppentq_l4'] = fqsm.groupby(['gvkey'])['ppentq'].shift(12)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(12)

fqsm['grltnoa'] = ((fqsm['rectq']+fqsm['invtq']+fqsm['ppentq']+fqsm['acoq']+fqsm['intanq']+
                       fqsm['aoq']-fqsm['apq']-fqsm['lcoq']-fqsm['loq'])-
                      (fqsm['rectq_l4']+fqsm['invtq_l4']+fqsm['ppentq_l4']+fqsm['acoq_l4']-fqsm['apq_l4']-fqsm['lcoq_l4']-fqsm['loq_l4'])-\
                     (fqsm['rectq']-fqsm['rectq_l4']+fqsm['invtq']-fqsm['invtq_l4']+fqsm['acoq']-
                      (fqsm['apq']-fqsm['apq_l4']+fqsm['lcoq']-fqsm['lcoq_l4'])-
                      ttm4('dpq', fqsm)))/((fqsm['atq']+fqsm['atq_l4'])/2)


# In[390]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','grltnoa']]
tmp.tail(20)


# In[391]:


#lev
fqsm['lev'] = fqsm['ltq']/fqsm['me']


# In[392]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','lev']]
tmp.tail(20)


# In[393]:


#lgr
fqsm['ltq_l4'] = fqsm.groupby(['gvkey'])['ltq'].shift(12)
fqsm['lgr'] = (fqsm['ltq']/fqsm['ltq_l4'])-1


# In[394]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','lgr']]
tmp.tail(20)


# In[395]:


#nincr
fqsm['ibq_l1'] = fqsm.groupby(['gvkey'])['ibq'].shift(3)
fqsm['ibq_l2'] = fqsm.groupby(['gvkey'])['ibq'].shift(6)
fqsm['ibq_l3'] = fqsm.groupby(['gvkey'])['ibq'].shift(9)
fqsm['ibq_l4'] = fqsm.groupby(['gvkey'])['ibq'].shift(12)
fqsm['ibq_l5'] = fqsm.groupby(['gvkey'])['ibq'].shift(15)
fqsm['ibq_l6'] = fqsm.groupby(['gvkey'])['ibq'].shift(18)
fqsm['ibq_l7'] = fqsm.groupby(['gvkey'])['ibq'].shift(21)
fqsm['ibq_l8'] = fqsm.groupby(['gvkey'])['ibq'].shift(24)

fqsm['nincr_temp1'] = np.where(fqsm['ibq'] > fqsm['ibq_l1'], 1, 0)
fqsm['nincr_temp2'] = np.where(fqsm['ibq_l1'] > fqsm['ibq_l2'], 1, 0)
fqsm['nincr_temp3'] = np.where(fqsm['ibq_l2'] > fqsm['ibq_l3'], 1, 0)
fqsm['nincr_temp4'] = np.where(fqsm['ibq_l3'] > fqsm['ibq_l4'], 1, 0)
fqsm['nincr_temp5'] = np.where(fqsm['ibq_l4'] > fqsm['ibq_l5'], 1, 0)
fqsm['nincr_temp6'] = np.where(fqsm['ibq_l5'] > fqsm['ibq_l6'], 1, 0)
fqsm['nincr_temp7'] = np.where(fqsm['ibq_l6'] > fqsm['ibq_l7'], 1, 0)
fqsm['nincr_temp8'] = np.where(fqsm['ibq_l7'] > fqsm['ibq_l8'], 1, 0)

fqsm['nincr'] = (fqsm['nincr_temp1']
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6']*fqsm['nincr_temp7'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6']*fqsm['nincr_temp7']*fqsm['nincr_temp8']))

fqsm = fqsm.drop(['ibq_l1', 'ibq_l2', 'ibq_l3', 'ibq_l4', 'ibq_l5', 'ibq_l6', 'ibq_l7', 'ibq_l8', 'nincr_temp1',
                            'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
                            'nincr_temp8'], axis=1)


# In[396]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','ibq','nincr']]
tmp.tail(20)


# In[397]:


#noa
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(12)
fqsm['ivaoq'] = np.where(fqsm['ivaoq'].isnull(), 0, 1)
fqsm['dlcq'] = np.where(fqsm['dlcq'].isnull(), 0, 1)
fqsm['dlttq'] = np.where(fqsm['dlttq'].isnull(), 0, 1)
fqsm['mibq'] = np.where(fqsm['mibq'].isnull(), 0, 1)
fqsm['pstkq'] = np.where(fqsm['pstkq'].isnull(), 0, 1)
fqsm['noa'] = (fqsm['atq']-fqsm['cheq']-fqsm['ivaoq'])-                 (fqsm['atq']-fqsm['dlcq']-fqsm['dlttq']-fqsm['mibq']-fqsm['pstkq']-fqsm['ceqq'])/fqsm['atq_l4']


# In[398]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','noa']]
tmp.tail(20)


# In[399]:


# op
fqsm['xintq'] = fqsm.groupby('gvkey')['xintq'].fillna(method='ffill')
fqsm['xsgaq'] = fqsm.groupby('gvkey')['xsgaq'].fillna(method='ffill')
fqsm['seqq'] = fqsm.groupby('gvkey')['seqq'].fillna(method='ffill')
fqsm['xintq0'] = np.where(fqsm['xintq'].isnull(), 0, fqsm['xintq'])
fqsm['xsgaq0'] = np.where(fqsm['xsgaq'].isnull(), 0, fqsm['xsgaq'])
fqsm['beq'] = np.where(fqsm['seqq']>0, fqsm['seqq']+0-fqsm['pstkq'], np.nan)
fqsm['beq'] = np.where(fqsm['beq']<=0, np.nan, fqsm['beq'])
fqsm['beq_l4'] = fqsm.groupby(['gvkey'])['beq'].shift(12)
fqsm['op'] = (ttm4('revtq', fqsm)-ttm4('cogsq', fqsm)-ttm4('xsgaq0', fqsm)-ttm4('xintq0', fqsm))/fqsm['beq_l4']


# In[400]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','seqq','pstkq','beq','op']]
tmp.tail(20)


# In[401]:


def mom(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['gvkey'])['retm'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result


# In[402]:


fqsm['mom12m'] = mom(1,12,fqsm)
fqsm['mom36m'] = mom(1,36,fqsm)
fqsm['mom60m'] = mom(12,60,fqsm)
fqsm['mom6m'] = mom(1,6,fqsm)
fqsm['mom1m'] = fqsm['retm']


# In[403]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','mom12m','mom36m','mom60m','mom6m','mom1m','retm']]
tmp.tail(20)


# In[404]:


#sgr
fqsm['saleq4'] = ttm4('saleq', fqsm)
fqsm['saleq4'] = np.where(fqsm['saleq4'].isnull(), fqsm['saley'], fqsm['saleq4'])
fqsm['saleq4_l4'] = fqsm.groupby(['gvkey'])['saleq4'].shift(12)
fqsm['sgr'] = (fqsm['saleq4']/fqsm['saleq4_l4'])-1


# In[405]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','sgr']]
tmp.tail(20)


# In[406]:


#ni
# fqsm['sps'] = fqsm['cshoc'] * fqsm['ajexdi']
# fqsm['sps_l1'] = fqsm.groupby('gvkey')['sps'].shift(3)
# fqsm['ni'] = np.log(fqsm['sps']/fqsm['sps_l1'])


# In[407]:


#rna
fqsm['oiadpq'] = fqsm.groupby('gvkey')['oiadpq'].fillna(method='ffill')
fqsm['noa_l4'] = fqsm.groupby(['gvkey'])['noa'].shift(12)
fqsm['rna'] = fqsm['oiadpq']/fqsm['noa_l4']


# In[408]:


#roa
fqsm['atq_l1'] = fqsm.groupby(['gvkey'])['atq'].shift(3)
fqsm['roa'] = fqsm['ibq']/fqsm['atq_l1']


# In[409]:


#roe
fqsm['ceqq_l1'] = fqsm.groupby(['gvkey'])['ceqq'].shift(3)
fqsm['roe'] = fqsm['ibq']/fqsm['ceqq_l1']


# In[410]:


#rsup
fqsm['saleq_l4'] = fqsm.groupby(['gvkey'])['saleq'].shift(12)
fqsm['rsup'] = (fqsm['saleq'] - fqsm['saleq_l4'])/fqsm['me']


# In[411]:


#seas1a
fqsm['seas1a'] = fqsm.groupby(['gvkey'])['retm'].shift(11)


# In[412]:


#sp
fqsm['sp'] = fqsm['saleq4']/fqsm['me']


# In[413]:


#acc
fqsm['iby'] = fqsm.groupby('gvkey')['iby'].fillna(method='ffill')
fqsm['oancfy'] = fqsm.groupby('gvkey')['oancfy'].fillna(method='ffill')
fqsm['acc'] = (fqsm['iby']-fqsm['oancfy'])/ttm4('atq',fqsm)


# In[414]:


# dy
# fqsm['me_l1'] = fqsm.groupby(['gvkey'])['me'].shift(3)
# fqsm['retdy'] = fqsm['retm'] - fqsm['retx']
# fqsm['mdivpay'] = fqsm['retdy']*fqsm['me_l1']


# In[415]:


#pctacc
fqsm['iby1'] = fqsm['iby'].replace(0,0.01)
fqsm['pctacc'] = (fqsm['iby']-fqsm['oancfy'])/abs(fqsm['iby1'])


# In[416]:


#pm
fqsm['pm'] = ttm4('oiadpq',fqsm)/ttm4('saleq',fqsm)


# In[417]:


tmp = fqsm[fqsm['gvkey']=='316100'][['gvkey','exchg','jdate','datadate_secm','datadate_fundq','saley','saleq','pm']]
tmp.tail(20)


# In[418]:


from tqdm import tqdm


# In[419]:


fqsm['date'] = fqsm.groupby(['gvkey'])['jdate'].shift(-1)


# In[420]:


def standardize(df):
    # exclude the the information columns
    col_names = df.columns.values.tolist()
    list_to_remove = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin',
       'datadate_secm','retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_fundq', 'pdateq',
       'fdateq', 'permno', 'jdate', 'date', 'datadate', 'sic', 'count', 'exchcd', 'shrcd', 'ffi49', 'ret',
       'retadj', 'retx', 'lag_me']
    col_names = list(set(col_names).difference(set(list_to_remove)))
    for col_name in tqdm(col_names):
        print('processing %s' % col_name)
        # count the non-missing number of factors, we only count non-missing values
        unique_count = df.dropna(subset=['%s' % col_name]).groupby(['date'])['%s' % col_name].unique().apply(len)
        unique_count = pd.DataFrame(unique_count).reset_index()
        unique_count.columns = ['date', 'count']
        df = pd.merge(df, unique_count, how='left', on=['date'])
        # ranking, and then standardize the data
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank(method='dense')
        df['rank_%s' % col_name] = (df['%s_rank' % col_name] - 1) / (df['count'] - 1) * 2 - 1
        df = df.drop(['%s_rank' % col_name, '%s' % col_name, 'count'], axis=1)
    df = df.fillna(0)
    return df


# In[421]:


df_rank = fqsm.copy()
df_rank['lag_me'] = df_rank['me']
df_rank = standardize(df_rank)


# In[422]:


df_rank.columns.values


# In[423]:


breakdown = df_rank.groupby(['jdate'])['rank_me'].describe(percentiles=[0.2,0.4,0.6,0.8]).reset_index()
breakdown = breakdown[['jdate','20%','40%','60%','80%']]


# In[424]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan
    
def plot3_a(col):
    charsa['test'] = charsa['rank_%s'%col]
    charsa['char_port'] = 1 + np.where(charsa['test']>-0.6,1,0) + np.where(charsa['test']>-0.2,1,0) + np.where(charsa['test']>0.2,1,0) + np.where(charsa['test']>0.6,1,0)
    vwret = charsa.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

    vwmkt = charsa.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
    vwmkt = vwmkt.reset_index()
    vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

    # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

    vwret = vwret[vwret['jdate'].dt.year>=2001]

    # figure 1 cumsum ret
    plt.figure(figsize=(15,5), dpi=80)
    plt.figure(1)
    plt.subplot(131)
    l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
    l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
    l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
    l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
    l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
    mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
    plt.title('rank_%s_a'%col)
    plt.legend()
    # figure 2
    # plt.figure(132)
    plt.subplot(132)
    plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
    plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
    plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
    plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
    plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
    plt.bar('mkt',vwret['vwret'].mean())
    plt.title('%s_avg_ret_a'%col)
    plt.legend()
    # figure 3
    # plt.figure(133)
    plt.subplot(133)
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
    plt.title('%s_num_a'%col)
    plt.legend()
    plt.savefig('./hkg2000/%s_a.jpg'%col)
#     plt.show()
    
def plot3_q(col):
    chars['test'] = chars['rank_%s'%col]
    chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)
    vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

    vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
    vwmkt = vwmkt.reset_index()
    vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

    # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

    vwret = vwret[vwret['jdate'].dt.year>=2001] # maybe modify

    # figure 1 cumsum ret
#     plt.cla()
    
    plt.figure(figsize=(15,5), dpi=80)
    plt.figure(1)
    plt.clf()
    plt.subplot(131)
    l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
    l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
    l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
    l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
    l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
    mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
    plt.title('rank_%s_q'%col)
    plt.legend()
    # figure 2
    # plt.figure(132)
    plt.subplot(132)
    plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
    plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
    plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
    plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
    plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
    plt.bar('mkt',vwret['vwret'].mean())
    plt.title('%s_avg_ret_q'%col)
    plt.legend()
    # figure 3
    # plt.figure(133)
    plt.subplot(133)
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
    plt.title('%s_num_q'%col)
    plt.legend()
    plt.savefig('./hkg2000/%s_q.jpg'%col) # change directory
    plt.close('all')
#     plt.show()


# In[ ]:


plotlist_q = ['bm', 'ep', 'cp', 'agr', 'alm', 'ato', 'cash', 'cashdebt', 
              'chpm', 'chtx', 'cinvest', 'depr', 'gma', 'grltnoa', 'lev', 
              'lgr', 'nincr', 'noa', 'op', 'mom12m', 'mom36m', 'mom60m', 
              'mom6m', 'mom1m', 'sgr', 'rna', 'roa', 'roe', 'rsup', 'seas1a', 'sp', 'acc', 'pctacc', 'pm']
for char in plotlist_q:
    plot3_q(char)


# In[179]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])
chars['test'] = chars['rank_bm']

chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
vwmkt = vwmkt.reset_index()
vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

# vwret['jdate'] = pd.to_datetime(vwret['jdate'])

vwret = vwret[vwret['jdate'].dt.year>=2001]

# figure 1 cumsum ret
plt.figure(figsize=(15,5), dpi=80)
plt.figure(1)
plt.subplot(131)
l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
plt.title('rank_bm')
plt.legend()
# figure 2
# plt.figure(132)
plt.subplot(132)
plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean())
plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean())
plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean())
plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean())
plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean())
plt.bar('mkt',vwret['vwret'].mean())
plt.title('bm_avg_ret')
plt.legend()
# figure 3
# plt.figure(133)
plt.subplot(133)
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_bm'].count(),label='port1')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_bm'].count(),label='port2')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_bm'].count(),label='port3')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_bm'].count(),label='port4')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_bm'].count(),label='port5')
plt.title('bm_num')
plt.legend()
plt.show()


# In[180]:


vwmkt.columns


# In[181]:


vwret[vwret['char_port']==3]['vwret'].mean()


# In[182]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])
chars['test'] = chars['rank_me']

chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
vwmkt = vwmkt.reset_index()
vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

# vwret['jdate'] = pd.to_datetime(vwret['jdate'])

vwret = vwret[vwret['jdate'].dt.year>=2001]

# figure 1 cumsum ret
plt.figure(figsize=(15,5), dpi=80)
plt.figure(1)
plt.subplot(131)
l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
plt.title('rank_me')
plt.legend()
# figure 2
# plt.figure(132)
plt.subplot(132)
plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
plt.bar('mkt',vwret['vwret'].mean())
plt.title('me_avg_ret')
plt.legend()
# figure 3
# plt.figure(133)
plt.subplot(133)
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_me'].count(),label='port1')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_me'].count(),label='port2')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_me'].count(),label='port3')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_me'].count(),label='port4')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_me'].count(),label='port5')
plt.title('me_num')
plt.legend()
plt.show()


# In[183]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])
chars['test'] = chars['rank_agr']

chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
vwmkt = vwmkt.reset_index()
vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

# vwret['jdate'] = pd.to_datetime(vwret['jdate'])

vwret = vwret[vwret['jdate'].dt.year>=2001]

# figure 1 cumsum ret
plt.figure(figsize=(15,5), dpi=80)
plt.figure(1)
plt.subplot(131)
l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
plt.title('rank_agr')
plt.legend()
# figure 2
# plt.figure(132)
plt.subplot(132)
plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
plt.bar('mkt',vwret['vwret'].mean())
plt.title('agr_avg_ret')
plt.legend()
# figure 3
# plt.figure(133)
plt.subplot(133)
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_agr'].count(),label='port1')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_agr'].count(),label='port2')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_agr'].count(),label='port3')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_agr'].count(),label='port4')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_agr'].count(),label='port5')
plt.title('agr_num')
plt.legend()
plt.show()


# In[184]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])
chars['test'] = chars['rank_op']

chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
vwmkt = vwmkt.reset_index()
vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

# vwret['jdate'] = pd.to_datetime(vwret['jdate'])

vwret = vwret[vwret['jdate'].dt.year>=2001]

# figure 1 cumsum ret
plt.figure(figsize=(15,5), dpi=80)
plt.figure(1)
plt.subplot(131)
l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
plt.title('rank_op')
plt.legend()
# figure 2
# plt.figure(132)
plt.subplot(132)
plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
plt.bar('mkt',vwret['vwret'].mean())
plt.title('op_avg_ret')
plt.legend()
# figure 3
# plt.figure(133)
plt.subplot(133)
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_op'].count(),label='port1')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_op'].count(),label='port2')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_op'].count(),label='port3')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_op'].count(),label='port4')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_op'].count(),label='port5')
plt.title('op_num')
plt.legend()
plt.show()


# In[185]:


chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])
chars['test'] = chars['rank_mom12m']

chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
vwmkt = vwmkt.reset_index()
vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

# vwret['jdate'] = pd.to_datetime(vwret['jdate'])

vwret = vwret[vwret['jdate'].dt.year>=2001]

# figure 1 cumsum ret
plt.figure(figsize=(15,5), dpi=80)
plt.figure(1)
plt.subplot(131)
l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
plt.title('rank_mom12m')
plt.legend()
# figure 2
# plt.figure(132)
plt.subplot(132)
plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
plt.bar('mkt',vwret['vwret'].mean())
plt.title('mom12m_avg_ret')
plt.legend()
# figure 3
# plt.figure(133)
plt.subplot(133)
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
plt.title('mom12m_num')
plt.legend()
plt.show()


# In[186]:


chars[(chars['jdate'].dt.year>2013) & (chars['jdate'].dt.year<=2020)].groupby(['jdate','char_port']).count()


# In[471]:


def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan
    
def plot3_a(col):
    charsa['test'] = charsa['rank_%s'%col]
    charsa['char_port'] = 1 + np.where(charsa['test']>-0.6,1,0) + np.where(charsa['test']>-0.2,1,0) + np.where(charsa['test']>0.2,1,0) + np.where(charsa['test']>0.6,1,0)
    vwret = charsa.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

    vwmkt = charsa.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
    vwmkt = vwmkt.reset_index()
    vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

    # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

    vwret = vwret[vwret['jdate'].dt.year>=2001]

    # figure 1 cumsum ret
    plt.figure(figsize=(15,5), dpi=80)
    plt.figure(1)
    plt.subplot(131)
    l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
    l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
    l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
    l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
    l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
    mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
    plt.title('rank_%s_a'%col)
    plt.legend()
    # figure 2
    # plt.figure(132)
    plt.subplot(132)
    plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
    plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
    plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
    plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
    plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
    plt.bar('mkt',vwret['vwret'].mean())
    plt.title('%s_avg_ret_a'%col)
    plt.legend()
    # figure 3
    # plt.figure(133)
    plt.subplot(133)
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
    plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
    plt.title('%s_num_a'%col)
    plt.legend()
    plt.savefig('./chn2000/%s_a.jpg'%col)
#     plt.show()
    
def plot3_q(col):
    chars['test'] = chars['rank_%s'%col]
    chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)
    vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

    vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
    vwmkt = vwmkt.reset_index()
    vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

    # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

    vwret = vwret[vwret['jdate'].dt.year>=2001] # maybe modify

    # figure 1 cumsum ret
    plt.figure(figsize=(15,5), dpi=80)
    plt.figure(1)
    plt.subplot(131)
    l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
    l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
    l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
    l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
    l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
    mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
    plt.title('rank_%s_q'%col)
    plt.legend()
    # figure 2
    # plt.figure(132)
    plt.subplot(132)
    plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
    plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
    plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
    plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
    plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
    plt.bar('mkt',vwret['vwret'].mean())
    plt.title('%s_avg_ret_q'%col)
    plt.legend()
    # figure 3
    # plt.figure(133)
    plt.subplot(133)
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
    plt.title('%s_num_q'%col)
    plt.legend()
    plt.savefig('./chn2000/%s_q.jpg'%col)
#     plt.show()


# In[187]:


chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me')


# In[188]:


fqsm[fqsm['jdate'].dt.year==2001]['bm'].isna().sum()
fqsm[fqsm['jdate'].dt.year==2020]['bm'].isna().sum()


# In[189]:


fqsm.columns.values


# In[190]:


tmp = fqsm[['date','me']]
tmp['me_na'] = tmp['me'].isna()
na_num = tmp.groupby('date').sum()['me_na']
tot_num = tmp.groupby('date').count()['me_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# # $bm = \frac{beq}{me}$

# In[191]:


tmp = fqsm[['date','bm']]
tmp['bm_na'] = tmp['bm'].isna()
na_num = tmp.groupby('date').sum()['bm_na']
tot_num = tmp.groupby('date').count()['bm_na']
# tmp


# In[192]:


tot_num


# In[193]:


na_num


# In[194]:


plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[195]:


fundq[['gvkey','datadate', 'pdateq', 'fdateq', 'fyr', 'ibq','jdate']].head()


# In[196]:


tmp = fundq[['datadate','beq']]
tmp['beq_na'] = tmp['beq'].isna()
na_num = tmp.groupby('datadate').sum()['beq_na']
tot_num = tmp.groupby('datadate').count()['beq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[197]:


tot_num


# In[198]:


na_num


# In[ ]:





# In[ ]:





# In[199]:


# fqsm['bm'].isna().groupby(['jdate'])


# In[200]:


plt.figure(1)
plt.plot(fqsm['jdate'],fqsm['bm'].isna(),label='bm')
plt.figure(2)
plt.plot(fqsm['jdate'],fqsm['me'].isna(),label='me')
plt.figure(3)
plt.plot(fqsm['jdate'],fqsm['agr'].isna(),label='agr')
plt.figure(4)
plt.plot(fqsm['jdate'],fqsm['mom12m'].isna(),label='mom12m')
plt.figure(5)
plt.plot(fqsm['jdate'],fqsm['op'].isna(),label='op')
plt.legend()
plt.show()


# In[201]:


plt.hist(chars[chars['jdate'].dt.year==2005]['rank_bm'])


# In[202]:


plt.hist(chars[chars['jdate'].dt.year==2018]['rank_bm'],bins=50)


# In[203]:


plt.hist(chars[chars['jdate'].dt.year==2014]['rank_bm'])


# In[204]:


plt.hist(fqsm[fqsm['jdate'].dt.year==2018]['bm'],bins=50)


# In[205]:


chars['year'] = chars['jdate'].dt.year


# In[206]:


fqsm['year']=fqsm['jdate'].dt.year


# In[207]:


fqsm['year']


# In[208]:


fqsm.tail(20)


# In[209]:


# fasm


# In[210]:


funda = pd.read_pickle('./funda_chn_2000.pkl')


# In[211]:


tmp = funda[['datadate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('datadate').sum()['seq_na']
tot_num = tmp.groupby('datadate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[212]:


tot_num


# In[213]:


# funda = funda[~funda['isin'].isna()]   # international id
# funda = funda[~funda['sedol'].isna()]   # international id
funda = funda[ (funda['exchg'] == 249) | (funda['exchg'] == 250)] # shanghai / shenzhen


# In[214]:


tmp = funda[['datadate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('datadate').sum()['seq_na']
tot_num = tmp.groupby('datadate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[215]:


funda.head(50)


# In[216]:


tot_num


# In[217]:


# funda = funda[~funda['pdate'].isna()] # some empty observations in fundq, you can check this with the next commented code
# fundq[fundq['gvkey']=='029530'].head(15)


# In[218]:


tmp = funda[['datadate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('datadate').sum()['seq_na']
tot_num = tmp.groupby('datadate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[219]:


funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol','seq'])


# In[220]:


tmp=funda.groupby(['gvkey','exchg']).nunique()['isin']
tmp[tmp>1]
# make sure, one gvkey-exchange has only one isin/sedol


# In[221]:


# print(funda.shape)
# # funda = funda[~funda['seq'].isna()]
# # funda = funda[~funda['ib'].isna()]
# print(funda.shape)


# In[222]:


funda['txdb'] = funda['txdb'].fillna(0)
funda['txt'] = funda['txt'].fillna(0)
funda['pstk'] = funda['pstk'].fillna(0)

funda['mib'] = funda['mib'].fillna(0)
funda['dlc'] = funda['dlc'].fillna(0)
funda['ivao'] = funda['ivao'].fillna(0)
funda['dltt'] = funda['dltt'].fillna(0)


# In[ ]:





# In[223]:


funda['be'] = funda['seq'] + funda['txdb'] + funda['txt'] - funda['pstk']


# In[224]:


funda.columns


# # Merge funda and secm

# In[225]:


# funda['datadate'] = pd.to_datetime(funda['datadate'])
# # join date is jdate
# # quarterly fundamentals are expected to report later than the datadate
# # 3 month is enough for the reporting process
# # thus, we don't have forseeing-data problem
# funda['jdate'] = funda['datadate'] + MonthEnd(3)
# funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol'])


# In[226]:


fundamental_varlist=[
    # id
    'gvkey', 'indfmt', 'consol', 'popsrc', 'datafmt','exchg', 'loc','fic', 'sedol', 'isin','datadate','pdate','fdate','fyr',
    # varaibles we want 
    'ib',
    'seq','txdb','txt','pstk','dp','at',
    'che','act','gdwl','intan','ceq',
    'ivao','dlc','dltt','mib','sale',
    'lt','ppent','revt','cogs',
    'rect','aco','ap','lco','lo','invt','ao','xint','xsga','be'
    ]
funda = funda[fundamental_varlist]
funda.head(50)


# In[227]:


funda['datadate'] = pd.to_datetime(funda['datadate'])
# join date is jdate
# quarterly fundamentals are expected to report later than the datadate
# 3 month is enough for the reporting process
# thus, we don't have forseeing-data problem
funda['jdate'] = funda['datadate'] + MonthEnd(6)
funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol'])


# In[228]:


tmp = funda[['datadate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('datadate').sum()['seq_na']
tot_num = tmp.groupby('datadate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[229]:


fasm = pd.merge(secm, funda, how='left', on=['gvkey','jdate','exchg','isin','sedol','loc','fic'])


# In[230]:


tmp = funda[['jdate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('jdate').sum()['seq_na']
tot_num = tmp.groupby('jdate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[231]:


tmp = fasm[fasm['gvkey']=='029530']
tmp.tail(20)


# # Forward Fill the Fundq info to Empty Month

# In[232]:


fasm.columns.values


# In[233]:


fasm.columns = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin',
       'datadate_secm', 'cshoc', 'prccd', 'me', 'retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_funda', 'pdate', 'fdate',
       'fyr', 'ib', 'seq', 'txdb', 'txt', 'pstk', 'dp', 'at', 'che',
       'act', 'gdwl', 'intan', 'ceq', 'ivao', 'dlc', 'dltt', 'mib',
       'sale', 'lt', 'ppent', 'revt', 'cogs', 'rect', 'aco', 'ap', 'lco',
       'lo', 'invt', 'ao', 'xint', 'xsga','be']


# In[234]:


fasm = fasm.sort_values(['gvkey','jdate','isin','sedol']) # order by gvkey, date, issue id, other id's


# In[235]:


fasm['pdate'] = fasm.groupby('gvkey')['pdate'].fillna(method='ffill')
fasm['fdate'] = fasm.groupby('gvkey')['fdate'].fillna(method='ffill')
fasm['ib'] = fasm.groupby('gvkey')['ib'].fillna(method='ffill')
fasm['be'] = fasm.groupby('gvkey')['be'].fillna(method='ffill')
# #


# In[236]:


fasm['me'] = fasm['me']/1e6 # 1e6 is one million


# In[237]:


fasm['bm'] = fasm['be']/fasm['me']
fasm['mb'] = fasm['me']/fasm['be']


# In[238]:


# op
fasm['xint'] = fasm.groupby('gvkey')['xint'].fillna(method='ffill')
fasm['xsga'] = fasm.groupby('gvkey')['xsga'].fillna(method='ffill')
fasm['seq'] = fasm.groupby('gvkey')['seq'].fillna(method='ffill')
fasm['revt'] = fasm.groupby('gvkey')['revt'].fillna(method='ffill')
fasm['cogs'] = fasm.groupby('gvkey')['cogs'].fillna(method='ffill')
fasm['xint0'] = np.where(fasm['xint'].isnull(), 0, fasm['xint'])
fasm['xsga0'] = np.where(fasm['xsga'].isnull(), 0, fasm['xsga'])
fasm['be'] = np.where(fasm['seq']>0, fasm['seq']+0-fasm['pstk'], np.nan)
fasm['be'] = np.where(fasm['be']<=0, np.nan, fasm['be'])
fasm['be'] = fasm.groupby('gvkey')['be'].fillna(method='ffill')
fasm['be_l4'] = fasm.groupby(['gvkey'])['be'].shift(12)
fasm['op'] = (ttm4('revt', fasm)-ttm4('cogs', fasm)-ttm4('xsga0', fasm)-ttm4('xint0', fasm))/fasm['be_l4']


# In[239]:


fasm['mom12m'] = mom(1,12,fasm)


# In[240]:


# agr
fasm['at'] = fasm.groupby('gvkey')['at'].fillna(method='ffill')
fasm['at_l4'] = fasm.groupby('gvkey')['at'].shift(12)
fasm['agr'] = (fasm['at'] - fasm['at_l4']) / fasm['at_l4']


# In[241]:


fasm[fasm['gvkey']=='029530'].tail(20)


# In[242]:


# new definition before
# def plot3_a(col):
#     charsa['test'] = charsa['rank_%s'%col]
#     charsa['char_port'] = 1 + np.where(charsa['test']>-0.6,1,0) + np.where(charsa['test']>-0.2,1,0) + np.where(charsa['test']>0.2,1,0) + np.where(charsa['test']>0.6,1,0)
#     vwret = charsa.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

#     vwmkt = charsa.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
#     vwmkt = vwmkt.reset_index()
#     vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

#     # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

#     vwret = vwret[vwret['jdate'].dt.year>=2001]

#     # figure 1 cumsum ret
#     plt.figure(figsize=(15,5), dpi=80)
#     plt.figure(1)
#     plt.subplot(131)
#     l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
#     l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
#     l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
#     l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
#     l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
#     mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
#     plt.title('rank_%s'%col)
#     plt.legend()
#     # figure 2
#     # plt.figure(132)
#     plt.subplot(132)
#     plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
#     plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
#     plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
#     plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
#     plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
#     plt.bar('mkt',vwret['vwret'].mean())
#     plt.title('%s_avg_ret'%col)
#     plt.legend()
#     # figure 3
#     # plt.figure(133)
#     plt.subplot(133)
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==1].groupby(['jdate'])['rank_mom12m'].count(),label='port1')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==2].groupby(['jdate'])['rank_mom12m'].count(),label='port2')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==3].groupby(['jdate'])['rank_mom12m'].count(),label='port3')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==4].groupby(['jdate'])['rank_mom12m'].count(),label='port4')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==5].groupby(['jdate'])['rank_mom12m'].count(),label='port5')
#     plt.title('%s_num'%col)
#     plt.legend()
#     plt.show()


# In[243]:


def standardize1(df):
    # exclude the the information columns
    col_names = df.columns.values.tolist()
    list_to_remove = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin',
       'datadate_secm','retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_funda', 'pdate',
       'fdate', 'permno', 'jdate', 'date', 'datadate', 'sic', 'count', 'exchcd', 'shrcd', 'ffi49', 'ret',
       'retadj', 'retx', 'lag_me','']
    col_names = list(set(col_names).difference(set(list_to_remove)))
    for col_name in tqdm(col_names):
        print('processing %s' % col_name)
        # count the non-missing number of factors, we only count non-missing values
        unique_count = df.dropna(subset=['%s' % col_name]).groupby(['date'])['%s' % col_name].unique().apply(len)
        unique_count = pd.DataFrame(unique_count).reset_index()
        unique_count.columns = ['date', 'count']
        df = pd.merge(df, unique_count, how='left', on=['date'])
        # ranking, and then standardize the data
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank(method='dense')
        df['rank_%s' % col_name] = (df['%s_rank' % col_name] - 1) / (df['count'] - 1) * 2 - 1
        df = df.drop(['%s_rank' % col_name, '%s' % col_name, 'count'], axis=1)
    df = df.fillna(0)
    return df


# In[244]:


fasm['date'] = fasm.groupby(['gvkey'])['jdate'].shift(-1)
df_ranka = fasm.copy()
df_ranka['lag_me'] = df_ranka['me']
df_ranka = standardize1(df_ranka)
charsa = df_ranka


# In[245]:


plot3_a('bm')


# In[246]:


plot3_a('me')


# In[247]:


plot3_a('agr')


# In[248]:


plot3_a('op')


# In[249]:


plot3_a('mom12m')


# In[250]:


tmp = fasm[['jdate','bm']]
tmp['bm_na'] = tmp['bm'].isna()
na_num = tmp.groupby('jdate').sum()['bm_na']
tot_num = tmp.groupby('jdate').count()['bm_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[251]:


tmp = funda[['datadate','seq']]
tmp['seq_na'] = tmp['seq'].isna()
na_num = tmp.groupby('datadate').sum()['seq_na']
tot_num = tmp.groupby('datadate').count()['seq_na']
# tmp
plt.figure()
plt.plot(tot_num,color='red')
plt.plot(na_num,color='blue')
plt.legend(['tot num','na num'])
plt.show()
plt.close()


# In[252]:


funda['fyear']


# In[ ]:




