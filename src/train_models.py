import keras
import yaml
from tensorflow import cast, float32
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.cluster import KMeans
from pandas import read_csv, concat, DataFrame, Series
import numpy as np
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import time
from custom_layers.rbf import RBFLayer
from emoji import EMOJIS

with open("../.config/config.yaml",'r') as fp:
    _config = yaml.safe_load(fp).get("train_models")

if _config.get("bypass",False):
    print("Bypassing...")
    exit()

print("Loading dataset...")
with open("../vol/data/combined.csv","r") as fp:
    dfX = read_csv(fp,index_col=0)
print("Done.")
dfX.info()

print("Separating target variables from dataset...")
dfLat = dfX["Latitude"]
dfLon = dfX["Longitude"]
dfX = dfX.drop(columns=["Latitude","Longitude"])
print("Done. Features: ", len(dfX.columns))

print("Splitting training and testing sets...")
dfX_train, dfX_test, \
dfLat_train, dfLat_test, \
dfLon_train, dfLon_test  = train_test_split(
    dfX,
    dfLat,
    dfLon,
    train_size=0.6,
    random_state=12345
)

print("Preparing dataset for training...")
feature_scale = MinMaxScaler((0,1))
lat_scale = MinMaxScaler((-1,1))
lon_scale = MinMaxScaler((-1,1))
print("\tNormalizing features...")
dfX_train = DataFrame(feature_scale.fit_transform(dfX_train))
dfX_test = DataFrame(feature_scale.transform(dfX_test))
print(dfX_train.shape)
print("\tNormalizing latitude values...")
dfLat_train = DataFrame(lat_scale.fit_transform(DataFrame(dfLat_train)))
dfLat_test = DataFrame(lat_scale.transform(DataFrame(dfLat_test)))
#print(dfLat_train.shape)
print("\tNormalizing longitude values...")
dfLon_train = DataFrame(lon_scale.fit_transform(DataFrame(dfLon_train)))
dfLon_test = DataFrame(lon_scale.transform(DataFrame(dfLon_test)))
#print(dfLon_train.shape)
print("Done.")

#print("Starting feature selection for latitude...")
#lasso_lat = sklearn.linear_model.LassoCV(cv=5).fit(dfX_train,dfLat_train)
#coefs_lat = Series(lasso_lat.coef_,index=dfX_train.columns)
#lat_feats = list(coefs_lat[coefs_lat != 0].index)
#print("Done. ",lat_feats)

#print("Starting feature selection for longitude...")
#lasso_lon = sklearn.linear_model.LassoCV(cv=5).fit(dfX_train,dfLon_train)
#coefs_lon = Series(lasso_lon.coef_,index=dfX_train.columns)
#lon_feats = list(coefs_lon[coefs_lon != 0].index)
#print("Done. ",lon_feats)

## Feedforward
print("Constructing feed-forward neural network...")
print("\tInitializing model...")
ff = keras.models.Sequential(name="feedforward")
print("\tAdding input layer...")
ff.add(keras.layers.Input(shape=(len(dfX_train.columns),)))
print("\tAdding hidden layer 1...")
ff.add(keras.layers.Dense(20,activation='relu',kernel_initializer='normal'))
print("\tAdding hidden layer 2...")
ff.add(keras.layers.Dense(10,activation='relu',kernel_initializer='normal',kernel_regularizer=keras.regularizers.l1(0.1)))
print("\tAdding output layer...")
ff.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal',kernel_regularizer=keras.regularizers.l1(0.1)))
print("\tCompiling model...")
ff.compile(
    optimizer="adam",
    loss="mse",
    metrics=["root_mean_squared_error","r2_score"]
)
print("Done.")
ff.summary()

## Recurrent
print("Constructing recurrent neural network...")
print("\tInitializing model...")
rnn = keras.models.Sequential(name="recurrent")
print("\tAdding input layer...")
rnn.add(keras.layers.Input(shape=(len(dfX_train.columns),1,)))
print("\tAdding hidden layer 1...")
rnn.add(keras.layers.LSTM(15,activation='relu',return_sequences=True))
print("\tAdding hidden layer 2...")
rnn.add(keras.layers.Dense(10,activation='relu',kernel_initializer='normal'))
print("\tAdding output layer...")
rnn.add(keras.layers.Dense(1,activation='linear'))
print("\tCompiling model...")
rnn.compile(
    optimizer="adam",
    loss="mse",
    metrics=["root_mean_squared_error","r2_score"]
)
print("Done.")
rnn.summary()

## Radial Basis Function
## code taken from https://stackoverflow.com/questions/53855941/how-to-implement-rbf-activation-function-in-keras
## citation: today. (December 20, 2018). How to implement RBF activation function in Keras? Stack Overflow. https://stackoverflow.com/a/53867101
print("Constructing radial basis function neural network...")

print("  Initializing model...")
rbf = keras.models.Sequential(name="radial_basis")
print("  Adding input layer...")
rbf.add(keras.layers.Input(shape=(len(dfX_train.columns),)))
print("  Adding custom RBF layer...")
rbf.add(RBFLayer(20,0.5))
print("  Adding hidden layer...")
rbf.add(keras.layers.Dense(10,activation='relu',kernel_initializer='normal'))
print("  Adding output layer...")
rbf.add(keras.layers.Dense(1,activation='linear'))
print("  Compiling model...")
rbf.compile(
    optimizer="adam",
    loss='mse',
    metrics=['root_mean_squared_error','r2_score']
)
print("Done.")
rbf.summary()

_info = [
    {"Model Name":"Feedforward","Model Type":"Neural Network",'Shorthand':'FFW'},
    {"Model Name":"Recurrent","Model Type":"Neural Network",'Shorthand':'REC'},
    {"Model Name":"Radial Basis Function","Model Type":"Neural Network",'Shorthand':'RBF'},
]

df_results = DataFrame()

dfLat_pred = {}
dfLon_pred = {}

thresh = int(len(dfX_train)/2)

if _config.get("FIT_NN",False):
    print("Training neural networks...")
    for model, info in zip([ff,rnn,rbf],_info):
        if _config.get("FIT_MODELS",{}).get(info.get("Shorthand","na"),False):
            print("\t",info['Model Name'],"...")
            _lat = DataFrame(lat_scale.inverse_transform(dfLat_test),columns = ['True Values'])
            _lon = DataFrame(lon_scale.inverse_transform(dfLon_test),columns = ['True Values'])
            for _iter in range(_config.get("ITERATIONS",1)):
                print(f"\t\tLatitude {_iter}...")
                start = time.time()
                history = model.fit(
                    cast(dfX_train.iloc[:thresh],float32), # first half
                    cast(dfLat_train.iloc[:thresh],float32), # first half
                    batch_size=128,
                    epochs=10,
                    validation_data=(
                        cast(dfX_train.iloc[thresh:],float32),
                        cast(dfLat_train.iloc[thresh:],float32)
                    ), # secong half
                    verbose=False,
                    callbacks = keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=3,
                        restore_best_weights=True
                    )
                )
                stop = time.time()
                print(EMOJIS['done'], "Done...",stop-start)

                _train = dict(
                    zip(
                        [
                            "Mean Square Error (Train)",
                            "Root MSE (Train)",
                            "R2 Score (Train)",
                            "Mean Square Error (Validate)",
                            "Root MSE (Validate)",
                            "R2 Score (Validate)"
                        ],
                        [x[-1] for x in history.history.values()]
                    )
                )

                #_eval = model.evaluate(dfX_test,dfLat_test,verbose=False)
                _pred = model.predict(dfX_test,verbose=False)
                _pred = np.expand_dims(_pred,axis=-1) if _pred.ndim==2 else _pred
                _pred = lat_scale.inverse_transform(DataFrame(_pred[:,-1,:]))
                _lat[f'Iter {_iter}'] = _pred

                _real = lat_scale.inverse_transform(DataFrame(dfLat_test))

                _result = {
                    'Iteration': _iter,
                    "Mean Square Error (Test)":round(mean_squared_error(_real,_pred),4),
                    "Root MSE (Test)":round(root_mean_squared_error(_real,_pred),4),
                    "R2 Score (Test)":round(r2_score(_real,_pred),4),
                    'Training Time': round(stop - start,4),
                    'Target Variable': "Latitude",
                    #'Feature Selection': False,
                }
                    
                print("\t\tAdding latitude results to dataframe...")
                df_results = concat([df_results,DataFrame([{**info,**_result,**_train}])])
                print(EMOJIS['done']," Done.")

                print(f"\t\tLongitude  {_iter}...")
                start = time.time()
                history = model.fit(
                    cast(dfX_train.iloc[:thresh],float32), # first half
                    cast(dfLon_train.iloc[:thresh],float32), # first half
                    batch_size=128,
                    epochs=10,
                    validation_data=(
                        cast(dfX_train.iloc[thresh:],float32),
                        cast(dfLon_train.iloc[thresh:],float32)
                    ), # second half
                    verbose=False,
                    callbacks = keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=3,
                        restore_best_weights=True
                    )
                )
                stop = time.time()
                print(EMOJIS['done'], " Done...",stop-start)

                _train = dict(zip(["Mean Square Error (Train)","Root MSE (Train)","R2 Score (Train)","Mean Square Error (Validate)","Root MSE (Validate)","R2 Score (Validate)"],[x[-1] for x in history.history.values()]))

                _pred = model.predict(dfX_test,verbose=False)
                _pred = np.expand_dims(_pred,axis=-1) if _pred.ndim==2 else _pred
                _pred = lon_scale.inverse_transform(DataFrame(_pred[:,-1,:]))
                _real = lon_scale.inverse_transform(DataFrame(dfLon_test))

                _lon[f'Iter {_iter}'] = _pred

                _result = {
                    'Iteration': _iter,
                    "Mean Square Error (Test)":round(mean_squared_error(_real,_pred),4),
                    "Root MSE (Test)":round(root_mean_squared_error(_real,_pred),4),
                    "R2 Score (Test)":round(r2_score(_real,_pred),4),
                    'Training Time': round(stop - start,4),
                    'Target Variable': "Longitude",
                    #'Feature Selection': False,
                }
                    
                print("\t\tAdding longitude results to dataframe...")
                df_results = concat([df_results,DataFrame([{**info,**_result,**_train}])])
                print(EMOJIS['done'], " Done.")

            dfLat_pred[info.get('Shorthand')] = _lat
            dfLon_pred[info.get('Shorthand')] = _lon

    print(EMOJIS['done']," Done.")

    print(EMOJIS['done']," Neural network training complete!")
else:
    print("Bypassing neural network training...")
print("Moving on to traditional regression...")
if _config.get("FIT_LIN_REG",False):
    print("De-normalizing features for traditional regression...")
    dfX_train = feature_scale.inverse_transform(dfX_train)
    dfX_test = feature_scale.inverse_transform(dfX_test)
    print(EMOJIS['done']," Done.")

    print("Training regression models...")
    if _config.get("FIT_MODELS",{}).get("MLR",False):
        print("  MLR...")
        lin_reg = LinearRegression()
        _lat = DataFrame(lat_scale.inverse_transform(dfLat_test),columns = ['True Values'])
        _lon = DataFrame(lon_scale.inverse_transform(dfLon_test),columns = ['True Values'])
        for target, target_train, target_test, df_target in zip(["Latitude","Longitude"],[dfLat_train,dfLon_train],[dfLat_test,dfLon_test],[_lat,_lon]):
            print("    ",target,"...")
            for _iter in range(_config.get("ITERATIONS",1)):
                start = time.time()
                lin_reg.fit(dfX_train,target_train)
                stop = time.time()

                # train_pred = mlr.predict(dfX_train)

                _pred = lin_reg.predict(dfX_test)
                _pred = lat_scale.inverse_transform(_pred) if target=="Latitude" else lon_scale.inverse_transform(_pred)
                
                df_target[f'Iter {_iter}'] = _pred

                _result = {
                    'Iteration': _iter,
                    "Model Name": "MLR",
                    "Model Type": "Traditional",
                    "Mean Square Error (Test)":round(mean_squared_error(target_test,_pred),4),
                    "Root MSE (Test)": round(root_mean_squared_error(target_test,_pred),4),
                    "R2 Score (Test)": round(r2_score(target_test,_pred),4),
                    "Target Variable": target,
                    "Training Time":round(stop-start,4),
                    #"Feature Selection": False
                }

                print(f"    [{_iter}] Writing test metrics to dataframe...")
                df_results = concat([df_results,DataFrame([_result])])
        dfLat_pred['MLR'] = _lat
        dfLon_pred['MLR'] = _lon 

    if _config.get("FIT_MODELS",{}).get("PLY",False):
        print("  Multi-Variate Poly...")
        poly = PolynomialFeatures(degree=2,include_bias=False)
        dfX_train_poly = poly.fit_transform(dfX_train)
        dfX_test_poly = poly.transform(dfX_test)
        _lat = DataFrame(lat_scale.inverse_transform(dfLat_test),columns = ['True Values'])
        _lon = DataFrame(lon_scale.inverse_transform(dfLon_test),columns = ['True Values'])
        for target, target_train, target_test, df_target in zip(["Latitude","Longitude"],[dfLat_train,dfLon_train],[dfLat_test,dfLon_test],[_lat,_lon]):
            print("    ",target,"...")
            for _iter in range(_config.get("ITERATIONS",1)):
                start = time.time()
                lin_reg.fit(dfX_train_poly,target_train)
                stop = time.time()

                # train_pred = mlr.predict(dfX_train)

                _pred = lin_reg.predict(dfX_test_poly)
                _pred = lat_scale.inverse_transform(_pred) if target=="Latitude" else lon_scale.inverse_transform(_pred)

                df_target[f'Iter {_iter}'] = _pred


                _result = {
                    'Iteration': _iter,
                    "Model Name": "2-MVPR",
                    "Model Type": "Traditional",
                    "Mean Square Error (Test)":round(mean_squared_error(target_test,_pred),4),
                    "Root MSE (Test)": round(root_mean_squared_error(target_test,_pred),4),
                    "R2 Score (Test)": round(r2_score(target_test,_pred),4),
                    "Target Variable": target,
                    "Training Time":round(stop-start,4),
                    #"Feature Selection": False
                }

                print(f"    [{_iter}] Writing test metrics to dataframe...")
                df_results = concat([df_results,DataFrame([_result])])
        dfLat_pred['MLR'] = _lat
        dfLon_pred['MLR'] = _lon 

    if _config.get("FIT_MODELS",{}).get("RDG",False):
        print("  Ridge...")
        std_scale = StandardScaler()
        ridge = RidgeCV(alphas = np.logspace(-3,3,100),cv=5)
        dfX_train_std = std_scale.fit_transform(dfX_train)
        dfX_test_std = std_scale.transform(dfX_test)
        _lat = DataFrame(lat_scale.inverse_transform(dfLat_test),columns = ['True Values'])
        _lon = DataFrame(lon_scale.inverse_transform(dfLon_test),columns = ['True Values'])
        for target, target_train, target_test, df_target in zip(["Latitude","Longitude"],[dfLat_train,dfLon_train],[dfLat_test,dfLon_test],[_lat,_lon]):
            print("    ",target,"...")
            for _iter in range(_config.get("ITERATIONS",1)):
                start = time.time()
                ridge.fit(dfX_train_std,target_train)
                stop = time.time()

                _pred = ridge.predict(dfX_test_std)
                _pred = lat_scale.inverse_transform(_pred) if target=="Latitude" else lon_scale.inverse_transform(_pred)

                df_target[f'Iter {_iter}'] = _pred

                _result = {
                    'Iteration': _iter,
                    "Model Name": "Ridge",
                    "Model Type": "Traditional",
                    "Mean Square Error (Test)":round(mean_squared_error(target_test,_pred),4),
                    "Root MSE (Test)": round(root_mean_squared_error(target_test,_pred),4),
                    "R2 Score (Test)": round(r2_score(target_test,_pred),4),
                    "Target Variable": target,
                    "Training Time":round(stop-start,4),
                    #"Feature Selection": False
                }

                print(f"    [{_iter}] Writing test metrics to dataframe...")
                df_results = concat([df_results,DataFrame([_result])])
        dfLat_pred['RDG'] = _lat
        dfLon_pred['RDG'] = _lon 
else:
    print("Bypassing traditional regression training...")

if df_results.empty:
    print("No results to write.")
else:
    print("Model training complete! Writing results to file...")
    with open("../vol/model/results.csv","w") as fp:
        df_results.to_csv(fp)
    print("Done.")

print("Writing predicted latitude values to file...")
for model,data in dfLat_pred.items():
    if not data.empty:
        with open(f"../vol/model/lat/{model.lower()}.csv","w") as fp:
            data.to_csv(fp)
print("Done.")

print("Writing predicted longitude values to file...")
for model,data in dfLon_pred.items():
    if not data.empty:
        with open(f"../vol/model/lon/{model.lower()}.csv","w") as fp:
            data.to_csv(fp)
print("Done.")