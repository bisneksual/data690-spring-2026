from pandas import DataFrame, read_csv, merge
from math import floor
from re import sub
import os
from datetime import datetime

WIND_DIRECTION = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

WEATHER_COLUMN_MAP = {
    "weather_code": "Weather Code",
    "temperature_2m_max":"Max Temp",
    "temperature_2m_min":"Min Temp",
    "apparent_temperature_mean":"Avg Apparent Temp",
    "daylight_duration":"Daylight",
    "precipitation_sum":"Precipitation",
    "wind_speed_10m_max":"Max Wind Sus",
    "wind_direction_10m_dominant":"Wind Bearing",
    "sunshine_duration":"Sunshine",
    "temperature_2m_mean":"Avg Temp",
    "wind_gusts_10m_max":"Max Wind Gusts",
    "et0_fao_evapotranspiration":"Evapotranspiration",
    "shortwave_radiation_sum":"Shortwave Radiation",
    "relative_humidity_2m_mean":"Avg Rel Humid",
    "relative_humidity_2m_max":"Max Rel Humid",
    "relative_humidity_2m_min":"Min Rel Humid",
    "pressure_msl_mean":"Avg Sea Level Pressure",
    "pressure_msl_max":"Max Sea Level Pressure",
    "pressure_msl_min":"Min Sea Level Pressure",
    "surface_pressure_mean":"Avg Surface Pressure",
    "surface_pressure_max":"Max Surface Pressure",
    "surface_pressure_min":"Min Surface Pressure",
    "cloud_cover_mean":"Avg Cloud Cover",
    "cloud_cover_max":"Max Cloud Cover",
    "cloud_cover_min":"Min Cloud Cover",
    "wet_bulb_temperature_2m_mean":"Avg Wet Bulb",
    "wet_bulb_temperature_2m_max":"Max Wet Bulb",
    "wet_bulb_temperature_2m_min":"Min Wet Bulb",
}

CRIME_CATEGORY = {
    "Identity Theft, Fraud, etc.": "354|666|662|664|951|654|649|940|652|653|950|660|651",
    "Assault": "230|625|647",
    "Theft From Vehicle": "331|420|410|421",
    "Crime Against Child": "812|760|235|814",
    "Neglect":"237|870",
    "Kidnapping": "910|922|920|434",
    "Theft": "440|480|350|442|351|444|670|474|446|471",
    "Theft, Attempt": "441|443|450|475|452|451|485|445",
    "Domestic Violence": "626|236",
    "Burglary": "310|330|320",
    "Battery": "624",
    "Vandalism, Trespassing": "745|740|888|924|432|882|926",
    "Stolen Vehicle": "510|520|522|487",
    "Sex Offenses": "860|956|122|821|121|810|820|815|762|840|830",
    "Grand Theft": "341|668|347|349|343|345|473",
    "Threats, Brandishing": "906|928",
    "Robbery": "210|220|353|453",
    "Violation of Court Order, etc": "900|901|903|845|902",
    "Arson":"648|755",
    "Other": "946|954|813|949|943|944|942|948",
    "Traffic Offenses": "890|438|433",
    "Misc Firearm": "753|250|251|931|756|904",
    "LEO Related Offenses": "623|231|437|439|622",
    "Obscenity, Indecency": "850|886|932|880",
    "Prostitution, Human Trafficking, etc.": "805|822|806|921",
    "Tech Related Crimes": "661",
    "Homicide": "110|113|435|436",
    "Drug Related Crimes": "865",
    "Stalking": "763|933",
}

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow fall",
    73: "Moderate snow fall",
    75: "Heacy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Mooderate rain showers",
    82: "Violent rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorn with heavy hail",
}

VICTIM_RACE_CODES = {
    "W":"White",
    "B": "Black",
    "I": "American Indian/Alaskan Native",
    "A": "Asian/Pacific Islander",
    "P": "Native Hawaiian/Other Pacific Islander",
    "H": "Hispanic or Latino",
    "U": "Unknown",
    "X": "Not Applicable",
}

#__logger = getLogger("data690.log")
#print(__logger.handlers)

#__logger.warning("Commencing data merging...")

#exit()
print("Checking for existing merged data...")
if os.path.exists("../vol/data/combined.csv"):
    print("Merged data found, bypassing merging script...")
    print("Done.")
    exit()

print("Done.")

print("Commencing data merging...")
print("Opening crime data...")
with open('../data/crime.csv','r') as fp_crime:
    df_crime = read_csv(fp_crime)
print("Done.")

print("Opening weather data...")
with open('../data/weather.csv','r') as fp_weather:
    df_weather = read_csv(fp_weather)
print("Done.")

print("Extracting date from report and occurrence timestamps...")
df_crime['Date Rptd'] = [x.split(' ')[0] for x in df_crime['Date Rptd']]
df_crime['DATE OCC'] = [x.split(' ')[0] for x in df_crime['DATE OCC']]
print("Done.")

print("Initializing crime dataframe...")
df_crime2 = DataFrame()
print("Done.")

print("Splitting report dates...")
df_crime2['Reported Year'] = [int(x.split('/')[-1]) for x in df_crime['Date Rptd']]
df_crime2['Reported Month'] = [int(x.split('/')[-3]) for x in df_crime['Date Rptd']]
df_crime2['Reported Day'] = [int(x.split('/')[-2]) for x in df_crime['Date Rptd']]
print("Done.")

print("Splitting occurrence dates...")
df_crime2['Occurrence Year'] = [int(x.split('/')[-1]) for x in df_crime['DATE OCC']]
df_crime2['Occurrence Month'] = [int(x.split('/')[-3]) for x in df_crime['DATE OCC']]
df_crime2['Occurrence Day'] = [int(x.split('/')[-2]) for x in df_crime['DATE OCC']]
print("Done.")

print("Assigning columns to crime dataframe...")
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

print("Done.")

print("Homogenizing crime dates for cross-referencing...")
df_crime2["Occurrence Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_crime2[['Occurrence Year','Occurrence Month','Occurrence Day']].itertuples(index=False,name=None)]
df_crime2["Report Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_crime2[['Reported Year','Reported Month','Reported Day']].itertuples(index=False,name=None)]
print("Done.")

print("Categorizing crime codes...")
df_crime2["Crime Category"] = [[key for key, val in CRIME_CATEGORY.items() if str(x) in val] + ["Other"] for x in df_crime2['Crime Code']]
df_crime2["Crime Category"] = [x[0] for x in df_crime2["Crime Category"]]
print("Done.")

#print(f"Dropping rows with non-acceptable victim race codes...{sum(1 if x not in VICTIM_RACE_CODES.keys() else 0 for x in df_crime2["Victim Race"])}")
#df_crime2 = df_crime2[[x in VICTIM_RACE_CODES.keys() for x in df_crime2["Victim Race"]]]
#print("Done")

print(f"Dropping rows with non-acceptable victim age values...{len([df_crime2["Victim Age"]<=0])}")
df_crime2 = df_crime2[df_crime2["Victim Age"]>0]
print("Done")

print("Categorizing victim race codes...")
df_crime2["V Race"] = [[val for key, val in VICTIM_RACE_CODES.items() if str(x) in key] + ["Unknown"] for x in df_crime2['Victim Race']]
df_crime2["Victim Race"] = [x[0] for x in df_crime2["V Race"]]
print("Done.")

print("Extracting date from weather timestamps...")
df_weather['date'] = [x.split(' ')[0] for x in df_weather['date']]
print("Done.")

df_weather2 = DataFrame()


print("Assigning columns to crime dataframe...")
for key, val in WEATHER_COLUMN_MAP.items():
    print("\t",val,"...")
    df_weather2[val] = df_weather[key]
print("Done.")

print("Splitting weather date...")
df_weather2['Year'] = [int(x.split('-')[0]) for x in df_weather['date']]
df_weather2['Month'] = [int(x.split('-')[-2]) for x in df_weather['date']]
df_weather2['Day'] = [int(x.split('-')[-1]) for x in df_weather['date']]
print("Done.")

print("Binning wind direction values...")
df_weather2['Wind Direction Index']  = [(floor((x / 22.5) + 0.5) % 16) for x in df_weather2['Wind Bearing']]
df_weather2['Wind Direction'] = [WIND_DIRECTION[x] for x in df_weather2['Wind Direction Index']]
print("Done.")

print("Mapping weather codes...")
df_weather2["Weather Code"] = [int(x) for x in df_weather2["Weather Code"]]
df_weather2["Weather Code"] = [WEATHER_CODE_MAP[x] for x in df_weather2["Weather Code"]]
print("Done.")

#print(df_weather2.head())

print("Homogenizing weather date for cross-referencing...")
df_weather2["Date"] = ["{:04d}{:02d}{:02d}".format(y, m, d) for y, m, d in df_weather2[['Year','Month','Day']].itertuples(index=False,name=None)]
print("Done.")

print("Merging crime and weather datasets using homogenized dates...")
df_combined = merge(df_crime2,df_weather2,left_on='Occurrence Date',right_on='Date',how='left')
print("Done.")

print("Re-establishing date format...")
df_combined["Date"] = [datetime.strptime(x,'%Y%m%d') for x in df_combined['Date']]
print("Done.")

print('Dropping report date...')
df_combined = df_combined.drop(columns=['Reported Year','Reported Month','Reported Day'])
print("Done.")

print('Dropping extranneous variables...')
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
        "Wind Direction Index",
        "Crime",
        "Crime Code",
        "V Race",
    ]
)

print("Dropping rows with missing values...")

print(f"\tLatitude...{len(df_combined[df_combined["Latitude"]==0])}")
df_combined = df_combined[df_combined["Latitude"]>0]

print(f"\tLongitude...{len(df_combined[df_combined["Longitude"]==0])}")
df_combined = df_combined[df_combined["Longitude"]<0]

print(df_combined["Victim Race"].value_counts())

print("Done.")

print("Writing combined dataset to file...")
with open("../vol/data/combined.csv","w") as fp:
    df_combined.to_csv(fp)
print("Done.")

print("Verifying that the file was written...",os.path.exists("../vol/data/combined.csv"))
print("Data merging complete!")

