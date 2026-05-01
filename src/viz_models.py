import yaml
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open("../.config/config.yaml","r") as fp:
    _config = yaml.safe_load(fp).get("viz_models",{})

if _config.get("bypass",False):
    print("Bypassing...")
    exit()

with open("../vol/model/results.csv","r") as fp:
    df_results = pd.read_csv(fp,index_col="Unnamed: 0")

df_results.info()
df_results['Shorthand'] = [x if isinstance(x,str) else "MLR" for x in df_results['Shorthand'].astype(str)]



for _name in df_results['Shorthand'].unique():
    time_box = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
    )
    for i, _target in enumerate(df_results['Target Variable'].unique()):
        _df = df_results[df_results['Shorthand'] == _name]
        _df = _df[_df['Target Variable'] == _target]
        _describe = _df.describe()
        #with open(f"../vol/model/viz/{_name.lower()}_{_target[:3].lower()}.yaml","w") as fp:
        #    yaml.safe_dump({'describe':_describe.to_dict()},fp,indent=2)
        with open(f"../vol/model/viz/{_name.lower()}_{_target[:3].lower()}.csv","w") as fp:
            _describe.to_csv(fp)

        time_box.add_trace(
            go.Box(
                x=_df['Training Time'],
                name=_name + "_" + _target[:3].lower()
            ),
            row=i+1,
            col=1,
        )
    time_box.write_html(f"../vol/model/viz/{_name.lower()}_time.html")