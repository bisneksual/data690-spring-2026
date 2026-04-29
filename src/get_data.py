import yaml
import pandas as pd
import openmeteo_requests
from retry_requests import retry
import requests_cache
import requests
import os
from emoji import EMOJIS

with open("../.config/config.yaml",'r') as fp:
    _config = yaml.safe_load(fp).get("get_data")

print(EMOJIS['file']," Checking for existing merged data...")
if os.path.exists('../vol/data/combined.csv'):
    print(EMOJIS['done'],"Merged data found, bypassing data fetching...")
    exit()
## Download Crime Data from portal

print(EMOJIS['file']," Checking for existing crime data...")
if os.path.exists('../data/crime.csv'):
    print(EMOJIS['done']," Crime data found, moving on...")
else:
    print(EMOJIS['wifi']," Fetching crime data from ",_config.get("CRIME_URL",""),"...")
    with requests.get(_config.get("CRIME_URL",""),stream=True) as r:
        r.raise_for_status()
        with open('../data/crime.csv','wb') as buf:
            for chunk in r.iter_content(chunk_size=8192):
                buf.write(chunk)
    print(EMOJIS['done']," Done.")

print(EMOJIS['file']," Checking for existing weather data...")
if os.path.exists('../data/weather.csv'):
    print(EMOJIS['done']," Weather data found, moving on...")
else:

    print(EMOJIS['wifi']," Fetching weather data from ",_config.get("WEATHER_URL",""),"...")

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below

    params = _config.get("WEATHER_PARAMS",{})
    responses = openmeteo.weather_api(_config.get("WEATHER_URL",""), params = params)

    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
        end =  pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    if False:
        for index, col in enumerate(_config.get('daily',[])):
            daily_data[col] = daily.Variables(index).ValuesAsNumpy()

    if True:
        daily_weather_code = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
        daily_apparent_temperature_mean = daily.Variables(3).ValuesAsNumpy()
        daily_daylight_duration = daily.Variables(4).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(5).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(6).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(7).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(8).ValuesAsNumpy()
        daily_temperature_2m_mean = daily.Variables(9).ValuesAsNumpy()
        daily_wind_gusts_10m_max = daily.Variables(10).ValuesAsNumpy()
        daily_et0_fao_evapotranspiration = daily.Variables(11).ValuesAsNumpy()
        daily_shortwave_radiation_sum = daily.Variables(12).ValuesAsNumpy()
        daily_relative_humidity_2m_mean = daily.Variables(13).ValuesAsNumpy()
        daily_relative_humidity_2m_max = daily.Variables(14).ValuesAsNumpy()
        daily_relative_humidity_2m_min = daily.Variables(15).ValuesAsNumpy()
        daily_pressure_msl_mean = daily.Variables(16).ValuesAsNumpy()
        daily_pressure_msl_max = daily.Variables(17).ValuesAsNumpy()
        daily_pressure_msl_min = daily.Variables(18).ValuesAsNumpy()
        daily_surface_pressure_mean = daily.Variables(19).ValuesAsNumpy()
        daily_surface_pressure_max = daily.Variables(20).ValuesAsNumpy()
        daily_surface_pressure_min = daily.Variables(21).ValuesAsNumpy()
        daily_cloud_cover_mean = daily.Variables(22).ValuesAsNumpy()
        daily_cloud_cover_max = daily.Variables(23).ValuesAsNumpy()
        daily_cloud_cover_min = daily.Variables(24).ValuesAsNumpy()
        daily_wet_bulb_temperature_2m_mean = daily.Variables(25).ValuesAsNumpy()
        daily_wet_bulb_temperature_2m_max = daily.Variables(26).ValuesAsNumpy()
        daily_wet_bulb_temperature_2m_min = daily.Variables(27).ValuesAsNumpy()
    
        daily_data["weather_code"] = daily_weather_code
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
        daily_data["daylight_duration"] = daily_daylight_duration
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
        daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
        daily_data["sunshine_duration"] = daily_sunshine_duration
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
        daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
        daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
        daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
        daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
        daily_data["relative_humidity_2m_min"] = daily_relative_humidity_2m_min
        daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
        daily_data["pressure_msl_max"] = daily_pressure_msl_max
        daily_data["pressure_msl_min"] = daily_pressure_msl_min
        daily_data["surface_pressure_mean"] = daily_surface_pressure_mean
        daily_data["surface_pressure_max"] = daily_surface_pressure_max
        daily_data["surface_pressure_min"] = daily_surface_pressure_min
        daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
        daily_data["cloud_cover_max"] = daily_cloud_cover_max
        daily_data["cloud_cover_min"] = daily_cloud_cover_min
        daily_data["wet_bulb_temperature_2m_mean"] = daily_wet_bulb_temperature_2m_mean
        daily_data["wet_bulb_temperature_2m_max"] = daily_wet_bulb_temperature_2m_max
        daily_data["wet_bulb_temperature_2m_min"] = daily_wet_bulb_temperature_2m_min

    daily_dataframe = pd.DataFrame(data = daily_data)

    with open('../data/weather.csv','w') as fp:
        daily_dataframe.to_csv(fp)
    print(EMOJIS['done']," Done.")
    print("Response Info: ")
    print(f"  {EMOJIS['pin']} Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"  {EMOJIS['mtn']} Elevation: {response.Elevation()} m asl")
    print(f"  {EMOJIS['clock']} Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"  {EMOJIS['clock']} Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
print(EMOJIS['end']," Data fetching complete!")