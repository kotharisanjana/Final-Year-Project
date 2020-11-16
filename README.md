# Lung cancer detection using deep learning and image processing

## How to train the model

1. Download the 10 subsets from luna 16 grand challenge website
2. Decompress them
3. Make 2 folders test, train
4. Copy the contents of subset9 to test
5. Copy the contents of subsets 1-9 to train
6. Download candidates_V2.csv
7. Preprocess data metadata ```python3 datasets.py```
8. Train the model. ```python3 train.py --batch_size 16/32/64```
9. Test the model. ```python3 test.py --model model/model_<last epoch> --batch_size 2/4/8/16```
10. Run the python notebook visualtest.ipynb to see the visualization of the dataset


## Files

### datasets.py
Loads the data, preprocesses it and creates a csv file containing normalized locations and image paths

### torch_ds.py
The dataset loader. It loads the scans dynamically when PyTorch needs it. It also preprocesses the data before sending it for training.

### torch_models.py
Contains the model architecture

### train.py
File to train the model. Calls torch_ds to load the data

### test.py
Runs metrics on the test data from the saved model

### visualtest.ipynb
Loads the models and visualizes the data vectors on a 2D plane
