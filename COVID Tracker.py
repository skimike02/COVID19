# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:38:33 2020

@author: Micha
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter    
import geopandas as gpd
import numpy as np
import math

state='CA'
counties=['Sacramento','El Dorado','Placer','Yolo']
fileloc=r'C:\\Users\Micha\Documents\GitHub\COVID19'

#Tests and National Stats
print("Getting national statistics...")
url='https://covidtracking.com/api/v1/states/daily.json'
df=pd.read_json(url)
df['Date']=pd.to_datetime(df.date, format='%Y%m%d', errors='ignore')
df=df[df['Date']>='2020-03-15']

def rolling_7_avg(df,date,group,field):
    newname=field+'_avg'
    df.sort_values(by=[group,date],inplace=True)
    df2=df.sort_values(by=[group,date]).assign(newname=df.groupby([group], as_index=False)[[field]].rolling(7,min_periods=7).mean().reset_index(0, drop=True))
    return df2.rename(columns={"newname": newname})

fields=['totalTestResultsIncrease','deathIncrease','positiveIncrease']


for field in fields:
    df=rolling_7_avg(df,'Date','state',field)

df['positivity']=df.positiveIncrease_avg/df.totalTestResultsIncrease_avg

#CA and County Stats
print("Getting California county statistics...")
#url='https://data.chhs.ca.gov/dataset/6882c390-b2d7-4b9a-aefa-2068cee63e47/resource/6cd8d424-dfaa-4bdd-9410-a3d656e1176e/download/covid19data.csv'
#dfca=pd.read_csv(url,delimiter=',')
#dfca['Date']=pd.to_datetime(dfca['Most Recent Date'], format='%m/%d/%Y')
#dfca['county']=dfca['County Name'].str.upper()

#CA Data
url='https://data.ca.gov/dataset/590188d5-8545-4c93-a9a0-e230f0db7290/resource/926fd08f-cc91-4828-af38-bd45de97f8c3/download/statewide_cases.csv'
caCases=pd.read_csv(url,delimiter=',')
url='https://data.ca.gov/dataset/529ac907-6ba1-4cb7-9aae-8966fc96aeef/resource/42d33765-20fd-44b8-a978-b083b7542225/download/hospitals_by_county.csv'
caHosp=pd.read_csv(url,delimiter=',')
caHosp = caHosp[pd.notnull(caHosp['todays_date'])]
caHosp = caHosp[pd.notnull(caHosp['county'])]
caData=caCases.merge(caHosp, how='left', left_on=['county','date'], right_on=['county','todays_date'])
caData['Date']=pd.to_datetime(caData['date'], format='%Y-%m-%d')
caData['COUNTY']=caData['county'].str.upper()
caData.rename(columns={'county':'County'}, inplace=True)
caData.drop(columns=['date','todays_date'], inplace=True)

#hospital capacity
print("Getting hospital capacity...")
url='https://data.chhs.ca.gov/datastore/dump/0997fa8e-ef7c-43f2-8b9a-94672935fa60?q=&sort=_id+asc&fields=FACID%2CFACNAME%2CFAC_FDR%2CBED_CAPACITY_TYPE%2CBED_CAPACITY%2CCOUNTY_NAME&filters=%7B%7D&format=csv'
df3=pd.read_csv(url,delimiter=',')
hospital_capacity=df3[df3['FAC_FDR']=='GENERAL ACUTE CARE HOSPITAL'].groupby('COUNTY_NAME').sum()['BED_CAPACITY']
ICU_capacity=df3[(df3['FAC_FDR']=='GENERAL ACUTE CARE HOSPITAL')&(df3['BED_CAPACITY_TYPE']=='INTENSIVE CARE')].groupby('COUNTY_NAME').sum()['BED_CAPACITY']
hospital_capacity.rename("hospital_capacity",inplace=True)
ICU_capacity.rename("ICU_capacity",inplace=True)
caData=caData.merge(hospital_capacity,left_on='COUNTY', right_index=True, how='left').merge(ICU_capacity,left_on='COUNTY', right_index=True, how='left')

#County Population
print("Getting county populations...")
url='https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv'
df4=pd.read_csv(url,delimiter=',',encoding='latin-1')
df4=df4[(df4['STATE']==6)&(df4['COUNTY']>0)]
df4['county']=df4['CTYNAME'].str.replace(' County','').str.upper()
df4=df4[['county','POPESTIMATE2019']]
caData=caData.merge(df4, left_on='COUNTY',right_on='county')
caData.rename(columns={"POPESTIMATE2019": "pop"},inplace=True)

"""
#Accelerated Reopening
print("Getting accelerated reopening plans...")
url='https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/County_Variance_Attestation_Form.aspx'
soup = bs(r.get(url).content, 'html.parser')
list=soup.findAll("div", {"class": "NewsItemContent"})[0].findAll("ul")[1].findAll("li")
accel_counties=[]
for item in list:
    for i in (item.findAll("a")[0].text.replace("County","").strip().split('-')):
        accel_counties.append(i)
"""

#County Data calculated fields
caData['hospitalized_confirmed_nonICU']=(caData['hospitalized_covid_confirmed_patients']-caData['icu_covid_confirmed_patients']).clip(0)
caData['hospitalized_suspected_nonICU']=(caData['hospitalized_suspected_covid_patients']-caData['icu_suspected_covid_patients']).clip(0)
caData['hospitalized']=caData['hospitalized_covid_patients']
caData['ICU']=caData['icu_covid_confirmed_patients']+caData['icu_suspected_covid_patients']
caData['ICU_usage']=caData['ICU']/caData['ICU_capacity']*100
caData['hospital_usage']=caData['hospitalized']/caData['hospital_capacity']*100
caData.sort_values(by=['county','Date'],inplace=True)
mask=~(caData.county.shift(1)==caData.county)
caData['positiveIncrease']=caData['newcountconfirmed'].clip(0)
caData['deathIncrease']=caData['newcountdeaths'].clip(0)
caData['noncovid_icu']=caData.ICU_capacity-caData.ICU-caData.icu_available_beds

fields=['positiveIncrease','deathIncrease']

for field in fields:
    caData=rolling_7_avg(caData,'Date','COUNTY',field)
    
fields=['positiveIncrease','deathIncrease','positiveIncrease_avg','deathIncrease_avg','hospitalized','ICU']
for field in fields:
    caData[field+'_percap']=caData[field]/caData['pop']*100000

regions=['US',state]
charts=['Tests','Cases','Deaths']

#%%
#Multichart US and CA
fig, axs = plt.subplots(len(charts),len(regions))
y=df.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean()
y2=df.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(1).mean()
x=df.Date.sort_values().unique()
axs[0, 0].plot(x, y2, color='grey', label='Daily')
axs[0, 0].plot(x, y, label='7-day avg')
axs[0, 0].set_title('US')
axs[0, 0].set(ylabel='Tests/Day')
axs[0, 0].legend(loc='upper left')

y=df.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean()
y2=df.groupby(['Date']).sum()['positiveIncrease'].rolling(1).mean()
x=df.Date.sort_values().unique()
axs[1, 0].plot(x, y2, color='grey', label='daily')
axs[1, 0].plot(x, y, label='7-day avg')
axs[1, 0].set(ylabel='Cases')

y=df.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean()
y2=df.groupby(['Date']).sum()['deathIncrease'].rolling(1).mean()
x=df.Date.sort_values().unique()
axs[2, 0].plot(x, y2, color='grey', label='daily')
axs[2, 0].plot(x, y, label='7-day avg')
axs[2, 0].set(ylabel='Deaths')

dfca=df[df['state']==state]
y=dfca.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean()
y2=dfca.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(1).mean()
x=dfca.Date.sort_values().unique()
axs[0, 1].plot(x, y, label='7 day average')
axs[0, 1].plot(x, y2, color='grey', label='daily')
axs[0, 1].set_title(state)

y=dfca.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean()
y2=dfca.groupby(['Date']).sum()['positiveIncrease'].rolling(1).mean()
x=dfca.Date.sort_values().unique()
axs[1, 1].plot(x, y, label='7 day average')
axs[1, 1].plot(x, y2, color='grey', label='daily')

y=dfca.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean()
y2=dfca.groupby(['Date']).sum()['deathIncrease'].rolling(1).mean()
x=dfca.Date.sort_values().unique()
axs[2, 1].plot(x, y, label='7 day average')
axs[2, 1].plot(x, y2, color='grey', label='daily')

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(b=None, which='major', axis='both')

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(11, 8.5)
plt.tight_layout()
fig.savefig("US_CA.png")
plt.show()



#%%
#Positivity US and CA
fig, axs = plt.subplots(1,len(regions))

y=100*df.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean()/df.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean()
x=df.Date.sort_values().unique()
axs[0].plot(x, y, label='7-day avg')
axs[0].set(ylabel='Positivity (7-day average)')
axs[0].set_ylim(0,30)
axs[0].set_title('US')

y=100*dfca.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean()/dfca.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean()
x=dfca.Date.sort_values().unique()
axs[1].plot(x, y, label='7 day average')
axs[1].set_ylim(0,30)
axs[1].set_title(state)

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(b=None, which='major', axis='both')

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(10, 5)
plt.tight_layout()
plt.show()

#%%
def countychart(county,df,data,percap):
    
    countydata=df[df['County']==county].sort_values(by=['Date'])
    countydata['new_cases']=countydata['newcountconfirmed']
    countydata['new_deaths']=countydata['newcountdeaths']
    countydata['14_day_cases']=countydata['totalcountconfirmed'].diff(periods=14)
    countydata['14_day_deaths']=countydata['totalcountdeaths'].diff(periods=14)
    x=countydata['Date']
    if percap:
        pop=countydata['pop'].values[0]/100000
    else:
        pop=1
    y=countydata[data].rolling(7).mean().rename('7-day avg')/pop
    y2=countydata[data].rename('Daily')/pop    
    return(x,y,y2)

#County Multichart
data=[]
#counties=['Sacramento','Solano']
#counties=['Sacramento','Placer','Amador','Butte','El Dorado','Lassen','Nevada']

#counties=['Sacramento','Sutter','Yuba','Modoc']
#counties=['Los Angeles','Orange','Riverside','San Diego']

for county in counties:
    data.append(countychart(county,caData,'new_cases',True))
for county in counties:
    data.append(countychart(county,caData,'new_deaths',False))
for county in counties:
    data.append((
    countychart(county,caData,'ICU_usage',False)[0],
    countychart(county,caData,'hospital_usage',False)[2].rename('Hospital usage'),
    countychart(county,caData,'ICU_usage',False)[2].rename('ICU usage')))

charts=['Cases','Deaths','Hospitalizations']
fig, axs = plt.subplots(len(charts),len(counties),sharex=True,sharey='row')
for ax, feature in zip(axs.flatten(), data):
    ax.plot(feature[0],feature[2],color='grey',label='Daily')
    ax.plot(feature[0],feature[1],label='7-day average')
    ax.set_ylim(bottom=0)
    ax.grid(b=None, which='major', axis='both')
    
axs[0, 0].set(ylabel='Cases/100k')
axs[0, 0].legend(loc='upper left')
axs[1, 0].legend(loc='upper left')
axs[1, 0].set(ylabel='Deaths')
axs[2, 0].set(ylabel='COVID-19 Hospital/ICU Usage %')
axs[2, 0].set_ylim(0,40)
axs[2, 0].legend(labels=['ICU','Hospitalization'],loc='upper left')
for ax,county in zip(axs[0],counties):
    ax.set_title(county)

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(11, 8.5)
fig.savefig("counties.png")
plt.tight_layout()
plt.show()

charts=['Cases']
fig, axs = plt.subplots(len(charts),len(counties),sharey='row')
for ax, feature in zip(axs.flatten(), data):
    ax.plot(feature[0],feature[2],color='grey',label='Daily')
    ax.plot(feature[0],feature[1],color='blue',label='7-day average')
    ax.set_ylim(bottom=0)
    ax.grid(b=None, which='major', axis='both')
    
axs[0].set(ylabel='Cases/100k')
axs[0].legend(loc='upper left')

for ax,county in zip(axs,counties):
    ax.set_title(county)

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(11, 3)
plt.tight_layout()
plt.show()

#%%
#CA Hospitalization
ca=caData.groupby(['Date']).sum().sort_values(by=['Date'])
ca=ca[ca.index>='2020-04-01']
y=[ca['hospitalized_confirmed_nonICU'],ca['hospitalized_suspected_nonICU'],ca['icu_covid_confirmed_patients'],ca['icu_suspected_covid_patients'],]

fig, ax = plt.subplots()
ax.stackplot(ca.index,y, labels=['hospitalized confirmed','hospitalized suspected','ICU confirmed','ICU suspected'])
ax.legend(loc="lower left")
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_title('CA Hospitalization Usage through '+ca.index.max().strftime('%Y-%m-%d'), size=22)
fig.autofmt_xdate(rotation=90)
fig.set_size_inches(11, 8.5)

fig.savefig("ca_hospitalization.png")
plt.show()


#%%
#County map chart
shp=fileloc+'\Counties\cb_2018_us_county_500k.shp'

map_df = gpd.read_file(shp)
caData.sort_values(by=['County','Date'],inplace=True)

def rolling_metric(metric_name,df_name,df):
    df[metric_name] = df[df_name].diff(14)
    mask = df['County'] != df['County'].shift(14)
    df[metric_name][mask] = np.nan
    return df

caData=rolling_metric('14_day_deaths','totalcountdeaths',caData)
caData['daily deaths/100k (14-day avg)']=caData['14_day_deaths']/caData['pop']/14*100000
caData=rolling_metric('14_day_cases','totalcountconfirmed',caData)
caData['daily cases/100k (14-day avg)']=caData['14_day_cases']/caData['pop']/14*100000
caData['i']=(caData.Date-caData[caData.Date>='2020-04-20'].Date.min()).dt.days

data=(('daily cases/100k (14-day avg)',0,5,'Reds','Daily cases/100k (14-day avg)'),
      ('daily deaths/100k (14-day avg)',0,0.3,'Reds','Daily deaths/100k (14-day avg)'),
      ('ICU_usage',0,30,'Reds','% ICU usage'),
      ('hospital_usage',0,30,'Reds','% Hospital Usage'))

fig, axs = plt.subplots(2,2, figsize=(15, 10))

def init():
    for ax, feature in zip(axs.flatten(), data):
        mapdata=caData[caData['i']==1]
        merged = map_df.merge(mapdata, how='inner', left_on="NAME", right_on="County")        # create map
        merged.plot(column='LSAD', linewidth=0.8, ax=ax, edgecolor='0.8')
        ax.axis('off')
        ax.set_title('', fontdict={'fontsize': '12', 'fontweight' : '3'})
    fig.suptitle('', y=0.92, fontsize=20)
    return fig   

def animate(i):
    mapdata=caData[caData['i']==i]
    merged = map_df.merge(mapdata, how='inner', left_on="NAME", right_on="County")
    #vmin, vmax = latest[variable].min(), latest[variable].max()
    
    for ax, feature in zip(axs.flatten(), data):
        # create map
        merged.plot(column=feature[0], cmap=feature[3],norm=plt.Normalize(vmin=feature[1], vmax=feature[2]), linewidth=0.8, ax=ax, edgecolor='0.8', legend=True if i==1 else "")
        ax.axis('off')
        ax.set_title(feature[4], fontdict={'fontsize': '12', 'fontweight' : '3'})
    fig.suptitle(mapdata.Date.max().strftime('%Y-%m-%d'), y=0.92, fontsize=20)
    return fig
    
anim = FuncAnimation(fig, animate, frames=caData.i.max()+1, interval=500)

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

#anim.save('animation.gif', writer=LoopingPillowWriter(fps=1))
animate(1)
latestmap=animate(caData.i.max())
latestmap.savefig("CountyMap.png")

#%%
"""
#Accelerated Counties
map_df['Accelerated']=map_df.NAME.isin(accel_counties)

if map_df['Accelerated'].value_counts()[1]!=len(accel_counties):
    print('error, counties do not match')
    for county in accel_counties:
        if county in set(map_df.NAME):
            continue
        else:
            print('unknown county: '+county)
 
fig, ax = plt.subplots(1, figsize=(8, 8))
map_df.plot(column='Accelerated', cmap='Greens', linewidth=0.8, edgecolor='0.8', ax=ax, legend=False)
ax.axis('off')
ax.set_title('Accelerated Counties', fontdict={'fontsize': '12', 'fontweight' : '3'})

fig.savefig("accelerated_counties.png")
"""
#%% state compare
fig, axs = plt.subplots(7,8)

i=0
for state2 in df.state.unique():
    dfca2=df[df['state']==state2]
    y=dfca2.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean()
    y2=dfca2.groupby(['Date']).sum()['deathIncrease'].rolling(1).mean()
    x=dfca2.Date.sort_values().unique()
    axs[math.trunc(i/8),i%8].plot(x, y, label='7 day average')
    axs[math.trunc(i/8),i%8].plot(x, y2, color='grey', label='daily')
    axs[math.trunc(i/8),i%8].set_title(state2,fontdict={'fontsize': '15', 'fontweight' : '3'})
    i=i+1

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(b=None, which='major', axis='both')

fig.suptitle('Deaths', y=0.995, fontsize=20)

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(25, 25)
plt.tight_layout()
plt.show()
#%%
"""

#Individual Charts


#Testing - US Daily
df.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean().plot(label='7 day average').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df.groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(1).mean().plot(color='grey', label='daily')
plt.title('US Daily Tests', size=22)
plt.ylabel('Number of Tests',size=18)
plt.xlabel('', size=18)
plt.legend(loc="upper left")
plt.grid(b=None, which='major', axis='y')
plt.show()

#Testing - CA 7 day average
df[df['state']=='CA'].groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(7).mean().plot(label='7 day average').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df[df['state']=='CA'].groupby(['Date']).sum()['totalTestResultsIncrease'].rolling(1).mean().plot(color='grey', label='daily')
plt.title('CA Daily Tests', size=22)
plt.ylabel('Number of Tests',size=18)
plt.xlabel('', size=18)
plt.legend(loc="upper left")
plt.grid(b=None, which='major', axis='y')
plt.show()

#Daily new positives US
#df.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean().plot()
df.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean().plot(label='7 day average').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df.groupby(['Date']).sum()['positiveIncrease'].rolling(1).mean().plot(color='grey', label='daily')
plt.title('US New Cases', size=22)
#plt.yscale('log')
plt.legend(loc="upper left")
plt.grid(b=None, which='major', axis='y')
plt.show()

#Daily new positives CA
#df.groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean().plot()
df[df['state']=='CA'].groupby(['Date']).sum()['positiveIncrease'].rolling(7).mean().plot(label='7 day average').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df[df['state']=='CA'].groupby(['Date']).sum()['positiveIncrease'].rolling(1).mean().plot(color='grey', label='daily')
plt.title('CA New Cases', size=22)
#plt.yscale('log')
plt.legend(loc="upper left")
plt.xlabel('', size=18)
plt.grid(b=None, which='major', axis='y')
plt.show()

#Daily new deaths US
#df.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean().plot()
df.groupby(['Date']).sum()['deathIncrease'].rolling(1).mean().plot(color='grey', label='daily').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean().plot(label='7 day average')
plt.title('US New Deaths', size=22)
plt.legend(loc="upper left")
plt.grid(b=None, which='major', axis='y')
plt.show()

#Daily new deaths CA
#df.groupby(['Date']).sum()['deathIncrease'].rolling(7).mean().plot()
df[df['state']=='CA'].groupby(['Date']).sum()['deathIncrease'].rolling(1).mean().plot(color='grey', label='daily').yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
df[df['state']=='CA'].groupby(['Date']).sum()['deathIncrease'].rolling(7).mean().plot(label='7 day average')
plt.title('CA New Deaths', size=22)
plt.legend(loc="upper left")
plt.xlabel('', size=18)
plt.grid(b=None, which='major', axis='y')
plt.show()


fig, ax = plt.subplots()
plt.title('CA Testing 7-day averages', size=22)
ca=df[df['state']=='CA']
plt.plot(ca.Date,ca.positiveIncrease.rolling(7).mean(),label='Positive')
plt.plot(ca.Date,ca.totalTestResultsIncrease.rolling(7).mean(),label='Tests')
plt.legend(loc="upper left")
plt.grid(b=None, which='major', axis='y')
fig.autofmt_xdate()
plt.show()

fig, ax = plt.subplots()
plt.plot(ca.Date,ca.positiveIncrease.rolling(7).mean()/ca.totalTestResultsIncrease.rolling(7).mean()*100,label='Positive Rate')
plt.title('CA Positive Rate', size=22)
plt.ylabel('% Positive',size=18)
plt.grid(b=None, which='major', axis='y')
fig.autofmt_xdate()
plt.show()



def countyplot(county,df2):
    countydata=df2[df2['County Name']==county].sort_values(by=['Date'])
    countydata['new_cases']=countydata['Total Count Confirmed'].diff()
    countydata['new_deaths']=countydata['Total Count Deaths'].diff()
    countydata['hospitalized_confirmed_nonICU']=(countydata['COVID-19 Positive Patients']-countydata['ICU COVID-19 Positive Patients']).clip(0)
    countydata['hospitalized_suspected_nonICU']=(countydata['Suspected COVID-19 Positive Patients']-countydata['ICU COVID-19 Suspected Patients']).clip(0)
    
    fig, axs = plt.subplots()
    plt.plot(countydata['Date'],countydata['new_cases'].rolling(7).mean(), label='7 day average')
    plt.plot(countydata['Date'],countydata['new_cases'],color='grey', label='Daily')
    plt.title(county+' New Cases', size=22)
    plt.legend(loc="upper left")
    plt.grid(b=None, which='major', axis='y')
    plt.ylim(bottom=0)
    fig.autofmt_xdate()
    plt.show()
    
    fig, ax = plt.subplots()
    plt.plot(countydata['Date'],countydata['new_deaths'].rolling(7).mean(), label='7 day average')
    plt.plot(countydata['Date'],countydata['new_deaths'],color='grey', label='Daily')
    plt.title(county+' New Deaths', size=22)
    plt.legend(loc="upper left")
    plt.grid(b=None, which='major', axis='y')
    plt.ylim(bottom=0)
    fig.autofmt_xdate()
    plt.show()

    y=[countydata['hospitalized_confirmed_nonICU'],countydata['hospitalized_suspected_nonICU'],countydata['ICU COVID-19 Positive Patients'],countydata['ICU COVID-19 Suspected Patients']] 
    fig, ax = plt.subplots()
    plt.stackplot(countydata['Date'],y, labels=['hospitalized confirmed','hospitalized suspected','ICU confirmed','ICU suspected'])
    plt.legend(loc="lower left")
    plt.title(county+' Hospitalization Usage', size=22)
    fig.autofmt_xdate()
    plt.show()


for county in counties:
    countyplot(county,df2)
    
positives=df[df.positive>=100].sort_values(by=['state','Date'])
positives['days_since']=positives.groupby(positives['state']).cumcount()+1

deaths=df[df.death>=100].sort_values(by=['state','Date'])
deaths['days_since']=deaths.groupby(deaths['state']).cumcount()+1

hospitalized=df[df.hospitalizedCurrently>=1].sort_values(by=['state','Date'])
hospitalized['days_since']=hospitalized.groupby(hospitalized['state']).cumcount()+1

#Cumulative Cases by State
chartdata=positives.pivot(index='days_since',columns='state', values='positive')
chartdata['Doubles every 4 days']=100*((2**(1/4))**chartdata.index)
chartdata['Doubles every 5 days']=100*((2**(1/5))**chartdata.index)
chartdata['Doubles every 6 days']=100*((2**(1/6))**chartdata.index)
chartdata.plot(color='grey')
chartdata.CA.plot(color='blue')
chartdata['Doubles every 4 days'].plot(color='grey')
chartdata['Doubles every 5 days'].plot(color='grey')
chartdata['Doubles every 6 days'].plot(color='grey')
plt.yscale('log')
plt.ylabel('Number of Cases',size=18)
plt.xlabel('Days since 100 Cases', size=18)
plt.title('CA Cumulative Cases', size=22)
plt.show()

#Cumulative Deaths by State
chartdata=deaths.pivot(index='days_since',columns='state', values='death')
chartdata['Doubles every 4 days']=100*((2**(1/4))**chartdata.index)
chartdata['Doubles every 5 days']=100*((2**(1/5))**chartdata.index)
chartdata['Doubles every 6 days']=100*((2**(1/6))**chartdata.index)
chartdata.plot(color='grey')
chartdata.CA.plot(color='blue')
chartdata['Doubles every 4 days'].plot(color='grey')
chartdata['Doubles every 5 days'].plot(color='grey')
chartdata['Doubles every 6 days'].plot(color='grey')
plt.yscale('log')
plt.ylabel('Number of Deaths',size=18)
plt.xlabel('Days since 100 Deaths', size=18)
plt.title('CA Cumulative Deaths', size=22)
plt.show()
"""