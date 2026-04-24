import keras
from pandas import read_csv, concat, DataFrame, Series
import matplotlib.pyplot as plt
import sklearn
import time

print("Loading dataset...")
with open("vol/data/combined.csv","r") as fp:
    dfX = read_csv(fp,index_col=0)
print("Done.")
dfX.info()

print("Separating target variables from dataset...")
dfLat = dfX["Latitude"]
dfLon = dfX["Longitude"]
dfX = dfX.drop(columns=["Latitude","Longitude"])
print("Done. Features: ", len(dfX.columns))

print("Preparing dataset for training...")
print("\tDropping string variables for now...")
dfX = dfX.select_dtypes(exclude="str")
#print("\tNormalizing latitude values...")
#dfLat = (dfLat - dfLat.min()) / (dfLat.max()-dfLat.min())
#print("\tNormalizing longitude values...")
#dfLon = (dfLon - dfLon.min())/(dfLon.max()-dfLon.min())
print("Done.")

print("Splitting training and testing sets...")
dfX_train, dfX_test, \
dfLat_train, dfLat_test, \
dfLon_train, dfLon_test  = sklearn.model_selection.train_test_split(
    dfX,
    dfLat,
    dfLon,
    train_size=0.6,
    random_state=12345
)

print("Splitting training and validation sets...")
thresh = int(len(dfX_train)/2)
print("\tX...")
dfX_valid = dfX_train.iloc[:thresh]
dfX_train = dfX_train.iloc[thresh:]
print("\tLatitude...")
dfLat_valid = dfLat_train.iloc[:thresh]
dfLat_train = dfLat_train.iloc[thresh:]
print("\tLongitude...")
dfLon_valid = dfLon_train.iloc[:thresh]
dfLon_train = dfLon_train.iloc[thresh:]
print("Done.")

print("Starting feature selection for latitude...")
lasso_lat = sklearn.linear_model.LassoCV(cv=5).fit(dfX_train,dfLat_train)
coefs_lat = Series(lasso_lat.coef_,index=dfX_train.columns)
lat_feats = list(coefs_lat[coefs_lat != 0].index)
print("Done. ",lat_feats)

print("Starting feature selection for longitude...")
lasso_lon = sklearn.linear_model.LassoCV(cv=5).fit(dfX_train,dfLon_train)
coefs_lon = Series(lasso_lon.coef_,index=dfX_train.columns)
lon_feats = list(coefs_lon[coefs_lon != 0].index)
print("Done. ",lon_feats)

## Feedforward
print("Constructing feed-forward neural network...")
print("\tInitializing model...")
ff = keras.models.Sequential(name="feedforward")
print("\tAdding input layer...")
ff.add(keras.layers.Input(shape=(30,)))
print("\tAdding hidden layer...")
ff.add(keras.layers.Dense(20,activation='relu',kernel_initializer='normal'))
print("\tAdding output layer...")
ff.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal'))
print("\tCompiling model...")
ff.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error","r2_score"]
)
print("Done.")
ff.summary()

## Recurrent
print("Constructing recurrent neural network...")
print("\tInitializing model...")
rnn_lat = keras.models.Sequential(name="recurrent")
print("\tCompiling model...")
rnn_lat.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error","r2_score"]
)
print("Done.")
rnn_lat.summary()

## Radial Basis Function
print("Constructing radial basis function neural network...")
print("\tInitializing model...")
rbf = keras.models.Sequential(name="radial_basis")
print("\tCompiling model...")
rbf.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["root_mean_squared_error","r2_score"]
)
print("Done.")
rbf.summary()

_info = [
    {"Model Name":"Feedforward","Model Type":"Neural Network"},
    {"Model Name":"Recurrent","Model Type":"Neural Network"},
    {"Model Name":"Radial Basis Function","Model Type":"Neural Network"},
]

df_results = DataFrame()

print("Training neural networks...")
for model, info in zip([ff,],_info):
    print("\t",info['Model Name'],"...")
    start = time.time()
    history = model.fit(
        dfX_train,
        dfLat_train,
        batch_size=128,
        epochs=10,
        validation_data=(dfX_valid,dfLat_valid),
        verbose=False,
        callbacks = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True
        )
    )
    stop = time.time()
    print(history.history)

    _result = dict(zip(["Mean Square Error (Test)","Root MSE (Test)","R2 Score (Test)"],model.evaluate(dfX_test,dfLat_test)))
    _result['Training Time'] = stop - start
    _result['Target Variable'] = "Latitude"
    _result['Feature Selection'] = False
        
    print("\t\tAdding latitude results to dataframe...")
    df_results = concat([df_results,DataFrame([{**info,**_result}])])

    start = time.time()
    history = model.fit(
        dfX_train,
        dfLon_train,
        batch_size=128,
        epochs=10,
        validation_data=(dfX_valid,dfLon_valid),
        verbose=False,
        callbacks = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True
        )
    )
    stop = time.time()
    print(history.history)

    _train = dict()

    _result = dict(zip(["Mean Square Error (Test)","Root MSE (Test)","R2 Score (Test)"],model.evaluate(dfX_test,dfLon_test)))
    _result['Training Time'] = stop - start
    _result['Target Variable'] = "Longitude"
    _result['Feature Selection'] = False
        
    print("\t\tAdding longitude results to dataframe...")
    df_results = concat([df_results,DataFrame([{**info,**_result}])])

print("Done.")

print("Neural network training complete! Moving on to traditional regression...")

## Multiple
print("Training multiple linear regression for latitude...")
mlr_lat = sklearn.linear_model.LinearRegression()
start = time.time()
mlr_lat.fit(concat([dfX_train,dfX_valid]),concat([dfLat_train,dfLat_valid]))
stop = time.time()
print("Done.")
_time = {"Target Variable":"Latitude","Training Time":stop-start}

## Polynomial

## Ridge

print("Model training complete! Writing results to file...")
with open("vol/model/results.csv","w") as fp:
    df_results.to_csv(fp)