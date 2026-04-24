import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

print("Commencing data exploration...")

print("Loading combined dataset...")
with open("data/combined.csv","r") as fp:
    df_combined = pd.read_csv(fp,index_col="Unnamed: 0")
print("Done.")

print("Loading weather dataset...")
with open("data/weather.csv","r") as fp:
    df_weather = pd.read_csv(fp,index_col="Unnamed: 0")
print("Done.")

print("Writing summary statistics to file...")
with open("explore/describe.json","w") as fp:
    df_combined.describe().to_json(fp,indent=4)
print("Done.")

print("Writing dataset info to file...")
with open("explore/info.json","w") as fp:
    df_combined.info(buf=fp)
print("Done.")

print("Rows: ",len(df_combined))
print("Columns: ",len(df_combined.columns))

#print("Writing crime code mapping to file...")
#with open("data/crime_map.json","w") as fp:
#    json.dump({key: val for key, val in zip(df_combined["Crime Code"],df_combined["Crime"])},fp,indent=4)
#print("Done.")

print("Writing crime category frequency table to file...")
with open("explore/crime_freq.json","w")  as fp:
    df_combined["Crime Category"].value_counts().to_json(fp,indent=4)
print("Done.")

print("Writing location frequency table to file...")
with open("explore/loc_freq.json","w")  as fp:
    df_combined["Location"].value_counts().to_json(fp,indent=4)
print("Done.")

print("Writing weather code frequency table to file...")
with open("explore/wmc_freq.json","w")  as fp:
    df_combined["Weather Code"].value_counts().to_json(fp,indent=4)
print("Done.")

print("Constructing visualizations...")
# Visualization 1
# Coordinate Heatmap

print("\tConstructing visualization 1...")
lat = df_combined['Latitude']
long = df_combined['Longitude']
coords = df_combined[["Longitude","Latitude"]]

#print("\t\tNormalizing latitude and longitude values...")
#norm_lat = [2 * (x - 34.0) / (34.5 - 34.0) - 1 for x in lat]
#norm_long = [2 * (x - (-118.75)) / (-118.25 - (-118.75)) - 1 for x in long]

#print(list(zip(norm_lat[:10],norm_long[:10])))

#print("\t\tGenerating heatmap values...")
#vals, x_edges, y_edges = np.histogram2d(norm_long,norm_lat,bins=np.linspace(-1.0,1.0,201))
#df = pd.DataFrame(vals)

#print(x_edges,"->",len(x_edges))
#print(y_edges,"->",len(y_edges))

df_coords = df_combined[["Latitude","Longitude"]]
coords = df_coords.value_counts().to_frame().reset_index()
#print(coords.head())
#print(coords.columns)

heatmap = px.density_map(
    coords,
    lon = 'Longitude',
    lat ='Latitude',
    z='count',
    center=dict(lat=34.2,lon=-118.5),
    radius=10,
    zoom=9,
)

print("\t\tWriting heatmap values to file...")
heatmap.write_json("vol/explore/coord_heatmap.json")
print("\t\tWriting heatmap to file...")
with open("vol/explore/coord_heatmap.html","w") as fp:
    heatmap.write_html(fp)

print("\tDone.")
quit()

# Visualization 2
# Crime Bar Chart Grouped By Area

print("\tConstructing visualization 2...")

crime_heatmap = px.density_heatmap(
    df_combined,
    x='Area',
    y='Crime Category',
)
print("\t\tWriting heatmap values to file...")
crime_heatmap.write_json("vol/explore/crime_heatmap.json")
print("\t\tWriting heatmap to file...")
with open("vol/explore/crime_heatmap.html","w") as fp:
    crime_heatmap.write_html(fp)

print("\tDone.")

# Visualization 3
# Time Series Line Chart

print("\tConstructing visualization 3...")

counts = df_combined["Date"].value_counts()
df_weather['date'] = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S%z").strftime("%Y-%m-%d") for x in df_weather['date']]

print(counts.head())
print(df_weather.head())

df_weather['crimes'] = pd.merge(
    left=pd.DataFrame({'Date':counts.index,"Count":counts.values}),
    left_on="Date",
    right=df_weather,
    right_on="date",
    how="left"
)["Count"]

#print(df_light.head())

weather_line = make_subplots(specs=[[{"secondary_y":True}]])

print("\t\tAdding weather lines...")
for label, arr in df_weather.items():
    if label != "crimes" and arr.dtype in [np.int64,np.float64,str]:
        weather_line.add_trace(
            go.Scatter(
                x=df_weather["date"],
                y=[(x-min(arr))/(max(arr)-min(arr)) for x in arr],
                name=label,
                line=dict(color='rgba(0, 0, 255, 0.2)',width=3)
            ),
            secondary_y=False
        )

print("\t\tAdding crime report line...")
weather_line.add_trace(
    go.Scatter(
        x=df_weather["date"],
        y=df_weather["crimes"],
        name="Crime Reports"
    ),
    secondary_y=True
)

print("\t\tUpdating chart title...")
weather_line.update_layout(
    title_text="Weather DenseLines Chart"
)
print("\t\tUpdating x axis title...")
weather_line.update_xaxes(title_text="Date")

print("\t\tUpdating y axes titles...")
weather_line.update_yaxes(title_text="Min-Max Value", secondary_y=False)
weather_line.update_yaxes(title_text="Number of Crime Reports", secondary_y=True)

print("\t\tWriting line chart to file...")
with open("vol/explore/weather_line.html","w") as fp:
    weather_line.write_html(fp)

print("\tDone")

print("Done.")