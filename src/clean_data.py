from pandas import read_csv, DataFrame
from datetime import datetime

df = None

try:
    with open('data/combined.csv','r') as fp:
        df = read_csv(fp)
except Exception as e:
    print(e)

if isinstance(df,DataFrame):
    # Drop unnecessary columns
    print("Dropping unnecessary columns...")
    df = df.drop(columns=['Unnamed: 0',"Occurrence Date","Year","Month","Day","Date"])

    df['Occurrence Date'] = [datetime(y,m,s).strftime("Y/m/d") for y,m,s in zip(df['Occurrence Year'],df['Occurrence Month'],df['Occurrence Day'])]
    
    print(df.head())