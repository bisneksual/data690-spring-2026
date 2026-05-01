mkdir data
mkdir vol/data
mkdir vol/explore
mkdir vol/model
mkdir vol/model/lat
mkdir vol/model/lon

rm -rf vol/model/viz
mkdir vol/model/viz

cd src

python3 -m get_data
python3 -m merge_data
python3 -m explore_data
python3 -m train_models
python3 -m viz_models