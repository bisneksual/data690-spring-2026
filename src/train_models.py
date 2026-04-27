import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas import read_csv, concat, DataFrame, Series
import numpy as np
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import time

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

print("Splitting training and validation sets...")
thresh = int(len(dfX_train)/2)
#print("\tX...")
#dfX_valid = dfX_train.iloc[:thresh]
#dfX_train = dfX_train.iloc[thresh:]
#print("\tLatitude...")
#dfLat_valid = dfLat_train.iloc[:thresh]
#dfLat_train = dfLat_train.iloc[thresh:]
#print("\tLongitude...")
#dfLon_valid = dfLon_train.iloc[:thresh]
#dfLon_train = dfLon_train.iloc[thresh:]
print("Done.")

print("Preparing dataset for training...")
feature_scale = MinMaxScaler((0,1))
target_scale = MinMaxScaler((-1,1))
print("\tNormalizing features...")
dfX_train = DataFrame(feature_scale.fit_transform(dfX_train))
dfX_test = DataFrame(feature_scale.transform(dfX_test))
print(dfX_train.shape)
print("\tNormalizing latitude values...")
dfLat_train = DataFrame(target_scale.fit_transform(DataFrame(dfLat_train)))
dfLat_test = DataFrame(target_scale.transform(DataFrame(dfLat_test)))
#print(dfLat_train.shape)
print("\tNormalizing longitude values...")
dfLon_train = DataFrame(target_scale.fit_transform(DataFrame(dfLon_train)))
dfLon_test = DataFrame(target_scale.transform(DataFrame(dfLon_test)))
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
ff.add(keras.layers.Input(shape=(30,)))
print("\tAdding hidden layer 1...")
ff.add(keras.layers.Dense(20,activation='relu',kernel_initializer='normal'))
print("\tAdding hidden layer 2...")
ff.add(keras.layers.Dense(10,activation='relu',kernel_initializer='normal'))
print("\tAdding output layer...")
ff.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal'))
print("\tCompiling model...")
ff.compile(
    optimizer="adam",
    loss=mean_squared_error,
    metrics=[root_mean_squared_error,r2_score]
)
print("Done.")
ff.summary()

## Recurrent
print("Constructing recurrent neural network...")
print("\tInitializing model...")
rnn = keras.models.Sequential(name="recurrent")
print("\tAdding input layer...")
rnn.add(keras.layers.Input(shape=(30,1,)))
print("\tAdding hidden layer 1...")
rnn.add(keras.layers.LSTM(10,activation='relu',return_sequences=True))
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
print("Constructing radial basis function neural network...")
print("\tRunning k-means clustering...")

print("\tInitializing model...")
rbf = keras.models.Sequential(name="radial_basis")
print("\tCompiling model...")
rbf.compile(
    optimizer="adam",
    loss=mean_squared_error,
    metrics=[root_mean_squared_error]
)
print("Done.")
rbf.summary()

_info = [
    #{"Model Name":"Feedforward","Model Type":"Neural Network"},
    {"Model Name":"Recurrent","Model Type":"Neural Network"},
    {"Model Name":"Radial Basis Function","Model Type":"Neural Network"},
]

df_results = DataFrame()

print("Training neural networks...")
for model, info in zip([rnn,],_info):
    print("\t",info['Model Name'],"...")
    start = time.time()
    history = model.fit(
        dfX_train.iloc[:thresh], # first half
        dfLat_train.iloc[:thresh], # first half
        batch_size=128,
        epochs=10,
        validation_data=(dfX_train.iloc[thresh:],dfLat_train.iloc[thresh:]), # secong half
        verbose=False,
        callbacks = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True
        )
    )
    stop = time.time()
    #print(history.history)

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
    _pred = target_scale.inverse_transform(DataFrame(_pred[:,-1,:]))
    _real = target_scale.inverse_transform(DataFrame(dfLat_test))

    _result = {
        "Mean Square Error (Test)":mean_squared_error(_real,_pred),
        "Root MSE (Test)":root_mean_squared_error(_real,_pred),
        "R2 Score (Test)":r2_score(_real,_pred),
        'Training Time': stop - start,
        'Target Variable': "Latitude",
        'Feature Selection': False,
    }
        
    print("\t\tAdding latitude results to dataframe...")
    df_results = concat([df_results,DataFrame([{**info,**_result,**_train}])])

    start = time.time()
    history = model.fit(
        dfX_train.iloc[:thresh], # first half
        dfLon_train.iloc[:thresh], # first half
        batch_size=128,
        epochs=10,
        validation_data=(dfX_train.iloc[thresh:],dfLon_train.iloc[thresh:]), # second half
        verbose=False,
        callbacks = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True
        )
    )
    stop = time.time()
    #print(history.history)

    _train = dict(zip(["Mean Square Error (Train)","Root MSE (Train)","R2 Score (Train)","Mean Square Error (Validate)","Root MSE (Validate)","R2 Score (Validate)"],[x[-1] for x in history.history.values()]))

    _pred = model.predict(dfX_test)
    _pred = target_scale.inverse_transform(DataFrame(_pred[:,-1,:]))
    _real = target_scale.inverse_transform(DataFrame(dfLon_test))

    _result = {
        "Mean Square Error (Test)":mean_squared_error(_real,_pred),
        "Root MSE (Test)":root_mean_squared_error(_real,_pred),
        "R2 Score (Test)":r2_score(_real,_pred),
        'Training Time': stop - start,
        'Target Variable': "Longitude",
        'Feature Selection': False,
    }
        
    print("\t\tAdding longitude results to dataframe...")
    df_results.loc[len(df_results)] = {**info,**_result,**_train}

print("Done.")

print("Neural network training complete! Moving on to traditional regression...")

## Multiple

mlr = LinearRegression()

print("Training regression models...")
for name, model in zip(["MLR","Poly","Ridge"],[mlr,]):
    print("\t",name,"...")
    for target, target_train, target_test in zip(["Latitude","Longitude"],[dfLat_train,dfLon_train],[dfLat_test,dfLon_test]):
        print("\t\t",target,"...")

        start = time.time()
        mlr.fit(dfX_train,target_train)
        stop = time.time()

        # train_pred = mlr.predict(dfX_train)

        _pred = mlr.predict(dfX_test)

        _result = {
            "Model Name": name,
            "Model Type": "Traditional",
            "Mean Square Error (Test)":mean_squared_error(target_test,_pred),
            "Root MSE (Test)": root_mean_squared_error(target_test,_pred),
            "R2 Score (Test)": r2_score(target_test,_pred),
            "Target Variable": target,
            "Training Time":stop-start,
            "Feature Selection": False
        }

        print("Writing test metrics to dataframe...")
        df_results.loc[len(df_results)] = _result

## Polynomial

## Ridge

print("Model training complete! Writing results to file...")
with open("../vol/model/results.csv","w") as fp:
    df_results.to_csv(fp)
print("Done.")