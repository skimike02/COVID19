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


state='CA'
counties=['Sacramento','El Dorado','Placer','Yolo']
fileloc=r'C:\\Users\Micha\Documents\GitHub\COVID19'

#Tests and National Stats
url='https://covidtracking.com/api/v1/states/daily.json'
df=pd.read_json(url)
df['Date']=pd.to_datetime(df.date, format='%Y%m%d', errors='ignore')
df=df[df['Date']>='2020-03-15']

#CA and County Stats
url='https://data.chhs.ca.gov/dataset/6882c390-b2d7-4b9a-aefa-2068cee63e47/resource/6cd8d424-dfaa-4bdd-9410-a3d656e1176e/download/covid19data.csv'
df2=pd.read_csv(url,delimiter=',')
df2['Date']=pd.to_datetime(df2['Most Recent Date'], format='%m/%d/%Y')
df2['county']=df2['County Name'].str.upper()

#hospital capacity
url='https://data.chhs.ca.gov/datastore/dump/0997fa8e-ef7c-43f2-8b9a-94672935fa60?q=&sort=_id+asc&fields=FACID%2CFACNAME%2CFAC_FDR%2CBED_CAPACITY_TYPE%2CBED_CAPACITY%2CCOUNTY_NAME&filters=%7B%7D&format=csv'
df3=pd.read_csv(url,delimiter=',')
hospital_capacity=df3[df3['FAC_FDR']=='GENERAL ACUTE CARE HOSPITAL'].groupby('COUNTY_NAME').sum()['BED_CAPACITY']
ICU_capacity=df3[(df3['FAC_FDR']=='GENERAL ACUTE CARE HOSPITAL')&(df3['BED_CAPACITY_TYPE']=='INTENSIVE CARE')].groupby('COUNTY_NAME').sum()['BED_CAPACITY']
hospital_capacity.rename("hospital_capacity",inplace=True)
ICU_capacity.rename("ICU_capacity",inplace=True)
df2=df2.merge(hospital_capacity,left_on='county', right_index=True, how='left').merge(ICU_capacity,left_on='county', right_index=True, how='left')

#County Population
url='https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv'
df4=pd.read_csv(url,delimiter=',',encoding='latin-1')
df4=df4[(df4['STATE']==6)&(df4['COUNTY']>0)]
df4['county']=df4['CTYNAME'].str.replace(' County','').str.upper()
df4=df4[['county','POPESTIMATE2019']]
df2=df2.merge(df4)
df2.rename(columns={"POPESTIMATE2019": "pop"},inplace=True)

#County Data calculated fields
df2['hospitalized_confirmed_nonICU']=(df2['COVID-19 Positive Patients']-df2['ICU COVID-19 Positive Patients']).clip(0)
df2['hospitalized_suspected_nonICU']=(df2['Suspected COVID-19 Positive Patients']-df2['ICU COVID-19 Suspected Patients']).clip(0)
df2['hospitalized']=df2['Suspected COVID-19 Positive Patients']+df2['COVID-19 Positive Patients']
df2['ICU']=df2['ICU COVID-19 Positive Patients']+df2['ICU COVID-19 Suspected Patients']
df2['ICU_usage']=df2['ICU']/df2['ICU_capacity']*100
df2['hospital_usage']=df2['hospitalized']/df2['hospital_capacity']*100

#url='https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'

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
axs[0, 0].set(ylabel='Tests')
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
def countychart(county,df2,data,percap):
    
    countydata=df2[df2['County Name']==county].sort_values(by=['Date'])
    countydata['new_cases']=countydata['Total Count Confirmed'].diff()
    countydata['new_deaths']=countydata['Total Count Deaths'].diff()
    countydata['14_day_cases']=countydata['Total Count Confirmed'].diff(periods=14)
    countydata['14_day_deaths']=countydata['Total Count Deaths'].diff(periods=14)
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
    data.append(countychart(county,df2,'new_cases',True))
for county in counties:
    data.append(countychart(county,df2,'new_deaths',False))
for county in counties:
    data.append((
    countychart(county,df2,'ICU_usage',False)[0],
    countychart(county,df2,'hospital_usage',False)[2].rename('Hospital usage'),
    countychart(county,df2,'ICU_usage',False)[2].rename('ICU usage')))

charts=['Cases','Deaths','Hospitalizations']
fig, axs = plt.subplots(len(charts),len(counties),sharex=True,sharey='row')
for ax, feature in zip(axs.flatten(), data):
    ax.plot(feature[0],feature[2],color='grey')
    ax.plot(feature[0],feature[1])
    ax.set_ylim(bottom=0)
    ax.grid(b=None, which='major', axis='both')
    
axs[0, 0].set(ylabel='Cases/100k')
axs[0, 0].legend(loc='upper left')
axs[1, 0].legend(loc='upper left')
axs[1, 0].set(ylabel='Deaths')
axs[2, 0].set(ylabel='Hospital/ICU Usage %')
axs[2, 0].set_ylim(0,100)
axs[2, 0].legend(loc='upper left')
for ax,county in zip(axs[0],counties):
    ax.set_title(county)

for ax in axs.flat:
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

fig.autofmt_xdate(rotation=90)
fig.set_size_inches(11, 8.5)
fig.savefig("counties.png")
plt.tight_layout()
plt.show()

#%%
#CA Hospitalization
ca=df2.groupby(['Date']).sum().sort_values(by=['Date'])
ca['hospitalized_confirmed_nonICU']=ca['COVID-19 Positive Patients']-ca['ICU COVID-19 Positive Patients']
ca['hospitalized_suspected_nonICU']=ca['Suspected COVID-19 Positive Patients']-ca['ICU COVID-19 Suspected Patients']
y=[ca['hospitalized_confirmed_nonICU'],ca['hospitalized_suspected_nonICU'],ca['ICU COVID-19 Positive Patients'],ca['ICU COVID-19 Suspected Patients'],]

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
shp=fileloc+'\Counties\cb_2018_us_county_500k.shp'

map_df = gpd.read_file(shp)
df2.sort_values(by=['County Name','Date'],inplace=True)

def rolling_metric(metric_name,df_name,df):
    df[metric_name] = df[df_name].diff(14)
    mask = df['County Name'] != df['County Name'].shift(14)
    df[metric_name][mask] = np.nan
    return df

df2=rolling_metric('14_day_deaths','Total Count Deaths',df2)
df2['daily deaths/100k (14-day avg)']=df2['14_day_deaths']/df2['pop']/14*100000
df2=rolling_metric('14_day_cases','Total Count Confirmed',df2)
df2['daily cases/100k (14-day avg)']=df2['14_day_cases']/df2['pop']/14*100000
df2['i']=(df2.Date-df2.Date.min()).dt.days-14

data=(('daily cases/100k (14-day avg)',1/14,5,'Reds','Daily cases/100k (14-day avg)'),
      ('daily deaths/100k (14-day avg)',0,0.3,'Reds','Daily deaths/100k (14-day avg)'),
      ('ICU_usage',0,30,'Reds','% ICU usage'),
      ('hospital_usage',0,30,'Reds','% Hospital Usage'))

fig, axs = plt.subplots(2,2, figsize=(15, 15))

def init():
    for ax, feature in zip(axs.flatten(), data):
        mapdata=df2[df2['i']==1]
        merged = map_df.merge(mapdata, how='inner', left_on="NAME", right_on="County Name")        # create map
        merged.plot(column='LSAD', linewidth=0.8, ax=ax, edgecolor='0.8')
        ax.axis('off')
        ax.set_title('', fontdict={'fontsize': '12', 'fontweight' : '3'})
    fig.suptitle('', y=0.92, fontsize=20)
    return fig   

def animate(i):
    mapdata=df2[df2['i']==i]
    merged = map_df.merge(mapdata, how='inner', left_on="NAME", right_on="County Name")
    #vmin, vmax = latest[variable].min(), latest[variable].max()
    
    for ax, feature in zip(axs.flatten(), data):
        # create map
        merged.plot(column=feature[0], cmap=feature[3],norm=plt.Normalize(vmin=feature[1], vmax=feature[2]), linewidth=0.8, ax=ax, edgecolor='0.8', legend=True if i==1 else "")
        ax.axis('off')
        ax.set_title(feature[4], fontdict={'fontsize': '12', 'fontweight' : '3'})
    fig.suptitle(mapdata.Date.max().strftime('%Y-%m-%d'), y=0.92, fontsize=20)
    return fig
    
anim = FuncAnimation(fig, animate, frames=df2.i.max()+1, interval=500)

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

#anim.save('animation.gif', writer=LoopingPillowWriter(fps=1))
animate(1)
latestmap=animate(df2.i.max())
latestmap.savefig("CountyMap.png")


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