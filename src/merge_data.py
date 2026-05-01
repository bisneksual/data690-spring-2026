from pandas import DataFrame, read_csv, merge
from math import floor
from emoji import EMOJIS
import os
from datetime import datetime
import yaml
import json

with open("../.config/config.yaml",'r') as fp:
    _config = yaml.safe_load(fp).get("merge_data")

if _config.get("bypass",False):
    print("Bypassing...")
    exit()

_race = _config.get("VICTIM_RACE_CODES",{})
_crime = _config.get("CRIME_CATEGORY",{})

#exit()
print(EMOJIS['file']," Checking for existing merged data...")
if os.path.exists("../vol/data/combined.csv"):
    print(EMOJIS['done']," Merged data found, bypassing merging script...")
    print(EMOJIS['end'],' Data merging complete!')
    exit()

print(EMOJIS['start']," Commencing data merging...")
print(EMOJIS['file']," Opening crime data...")
with open('../data/crime.csv','r') as fp_crime:
    df_crime = read_csv(fp_crime)
print(EMOJIS['done']," Done.")

print(EMOJIS['file']," Opening weather data...")
with open('../data/weather.csv','r') as fp_weather:
    df_weather = read_csv(fp_weather)
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Extracting date from report and occurrence timestamps...")
df_crime['Date Rptd'] = [x.split(' ')[0] for x in df_crime['Date Rptd']]
df_crime['DATE OCC'] = [x.split(' ')[0] for x in df_crime['DATE OCC']]
print(EMOJIS['done']," Done.")

print(EMOJIS['crime']," Initializing crime dataframe...")
df_crime2 = DataFrame()
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Splitting report dates...")
df_crime2['Reported Year'] = [int(x.split('/')[-1]) for x in df_crime['Date Rptd']]
df_crime2['Reported Month'] = [int(x.split('/')[-3]) for x in df_crime['Date Rptd']]
df_crime2['Reported Day'] = [int(x.split('/')[-2]) for x in df_crime['Date Rptd']]
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Splitting occurrence dates...")
df_crime2['Occurrence Year'] = [int(x.split('/')[-1]) for x in df_crime['DATE OCC']]
df_crime2['Occurrence Month'] = [int(x.split('/')[-3]) for x in df_crime['DATE OCC']]
df_crime2['Occurrence Day'] = [int(x.split('/')[-2]) for x in df_crime['DATE OCC']]
print(EMOJIS['data']," Done.")

print(EMOJIS['crime']," Assigning columns to crime dataframe...")
print("\tArea...")
df_crime2['Area'] = df_crime['AREA NAME']

print("\tCrime...")
df_crime2['Crime'] = df_crime['Crm Cd Desc']

print("\tCrime Code...")
df_crime2['Crime Code'] = df_crime['Crm Cd']

print("\tVictim Age...")
df_crime2['Victim Age'] = df_crime['Vict Age']

print("\tVictim Sex...")
df_crime2['Victim Sex'] = [x=="M" for x in df_crime['Vict Sex']]

print("\tVictim Race...")
df_crime2['Victim Race'] = df_crime['Vict Descent']

print("\tLocation...")
df_crime2['Location'] = df_crime['Premis Desc']

#print("\tAddress...")
#df_crime2['Address'] = [sub(r"/s+"," ",x).strip() for x in df_crime['LOCATION']]

print("\tIntersection...")
df_crime2['Intersection'] = [x=="" for x in df_crime['Cross Street']]

print("\tLatitude...")
df_crime2["Latitude"] = df_crime["LAT"]

print("\tLongitude...")
df_crime2["Longitude"] = df_crime["LON"]

print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Homogenizing crime dates for cross-referencing...")
df_crime2["Occurrence Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_crime2[['Occurrence Year','Occurrence Month','Occurrence Day']].itertuples(index=False,name=None)]
df_crime2["Report Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_crime2[['Reported Year','Reported Month','Reported Day']].itertuples(index=False,name=None)]
print(EMOJIS['done']," Done.")

print(EMOJIS['crime']," Categorizing crime codes...")
df_crime2["Crime Category"] = [[key for key, val in _crime.items() if str(x) in val] + ["Other"] for x in df_crime2['Crime Code']]
df_crime2["Crime Category"] = [x[0] for x in df_crime2["Crime Category"]]
print(EMOJIS['done']," Done.")

#print(f"Dropping rows with non-acceptable victim race codes...{sum(1 if x not in VICTIM_RACE_CODES.keys() else 0 for x in df_crime2["Victim Race"])}")
#df_crime2 = df_crime2[[x in VICTIM_RACE_CODES.keys() for x in df_crime2["Victim Race"]]]
#print(EMOJIS['data']," Done.")

print(EMOJIS['rm'],f" Dropping rows with non-acceptable victim age values...{len([df_crime2["Victim Age"]<=0])}")
df_crime2 = df_crime2[df_crime2["Victim Age"]>0]
print(EMOJIS['done']," Done.")

print(EMOJIS['crime'],"Categorizing victim race codes...")
df_crime2["V Race"] = [[val for key, val in _race.items() if str(x) in key] + ["Unknown"] for x in df_crime2['Victim Race']]
df_crime2["Victim Race"] = [x[0] for x in df_crime2["V Race"]]
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Extracting date from weather timestamps...")
df_weather['date'] = [x.split(' ')[0] for x in df_weather['date']]
print(EMOJIS['done']," Done.")

df_weather2 = DataFrame()


print(EMOJIS['weather']," Assigning columns to weather dataframe...")
print("WEATHER_COLUMN_MAP" in _config)
for key, val in _config.get("WEATHER_COLUMN_MAP",{}).items():
    print("\t",val,"...")
    df_weather2[val] = df_weather[key]
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Splitting weather date...")
df_weather2['Year'] = [int(x.split('-')[0]) for x in df_weather['date']]
df_weather2['Month'] = [int(x.split('-')[-2]) for x in df_weather['date']]
df_weather2['Day'] = [int(x.split('-')[-1]) for x in df_weather['date']]
print(EMOJIS['done']," Done.")

#print(EMOJIS['wind']," Binning wind direction values...")
#df_weather2['Wind Direction Index']  = [(floor((x / 22.5) + 0.5) % 16) for x in df_weather2['Wind Bearing']]
#df_weather2['Wind Direction'] = [WIND_DIRECTION[x] for x in df_weather2['Wind Direction Index']]
#print(EMOJIS['done']," Done.")

#print("Mapping weather codes...")
#df_weather2["Weather Code"] = [int(x) for x in df_weather2["Weather Code"]]
#df_weather2["Weather Code"] = [WEATHER_CODE_MAP[x] for x in df_weather2["Weather Code"]]
#print(EMOJIS['data']," Done.")

#print(df_weather2.head())

print(EMOJIS['date']," Homogenizing weather date for cross-referencing...")
df_weather2["Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_weather2[['Year','Month','Day']].itertuples(index=False,name=None)]
print(EMOJIS['done']," Done.")

print(EMOJIS['merge']," Merging crime and weather datasets using homogenized dates...")
df_combined = merge(df_crime2,df_weather2,left_on='Occurrence Date',right_on='Date',how='left')
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," Re-establishing date format...")
df_combined["Date"] = [datetime.strptime(x,'%Y%m%d') for x in df_combined['Date']]
print(EMOJIS['done']," Done.")

print(EMOJIS['date'],' Dropping report date...')
df_combined = df_combined.drop(columns=['Reported Year','Reported Month','Reported Day'])
print(EMOJIS['done']," Done.")

print(EMOJIS['rm'],' Dropping extranneous variables...')
df_combined = df_combined.drop(
    columns=[
        'Occurrence Year',
        "Occurrence Month",
        "Occurrence Day",
        "Year",
        "Month",
        "Day",
        "Occurrence Date",
        "Report Date",
        "Crime",
        "Crime Code",
        "V Race",
        "Location"
    ]
)

print(EMOJIS['rm']," Dropping rows with missing values...")

print(f"\tLatitude...{len(df_combined[df_combined["Latitude"]==0])}")
df_combined = df_combined[df_combined["Latitude"]>0]

print(f"\tLongitude...{len(df_combined[df_combined["Longitude"]==0])}")
df_combined = df_combined[df_combined["Longitude"]<0]

print(df_combined["Victim Race"].value_counts())

print(EMOJIS['done']," Done.")

#with open("../vol/explore/loc_vals.json","w") as fp:
#    json.dump(df_combined['Location'].value_counts().to_dict(),fp)

#with open("../vol/explore/_vals.json","w") as fp:
#    json.dump(df_combined['Location'].value_counts().to_dict(),fp)

print(EMOJIS['data']," Indexing string variables...")
for col, data in df_combined.items():
    if data.dtype == 'str':
        vals = list(data.unique())
        with open(f"../vol/explore/{col}.json","w") as fp:
            json.dump(dict(enumerate(vals)),fp)
        df_combined[col] = [vals.index(x) for x in df_combined[col]]
print(EMOJIS['done']," Done.")

print(EMOJIS['date']," converting date to days since epoch...")
df_combined['Date'] = [(x - datetime(1970,1,1)).days for x in df_combined['Date']]
print(EMOJIS['done']," Done.")

print(EMOJIS['write']," Writing combined dataset to file...")
with open("../vol/data/combined.csv","w") as fp:
    df_combined.to_csv(fp)
print(EMOJIS['done']," Done.")

print(EMOJIS['file']," Verifying that the file was written...",os.path.exists("../vol/data/combined.csv"))
print(EMOJIS['end'],"Data merging complete!")

