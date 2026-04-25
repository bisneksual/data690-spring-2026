mkdir vol/data
mkdir vol/explore
mkdir vol/model

cd src

python3 -m get_data
python3 -m merge_data
# python3 -m explore_data
python3 -m train_models