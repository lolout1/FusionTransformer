# Fall Detection KD Multimodal 
## Getting started 
- Create an conda environment and use the requirements.txt to install all the neccasary files.

## Get the Dataset

- Find all the datasets and npz files [here](). 
- Download the data and put it under the root directory. 
- The datasets can processed from the scartch using the Processing_data.ipynb file.

## Choose and configure models
- Model configuration for Accelerometer model is kept under ``config/utd/student.yaml`` and in ``config/utd/teacher.yaml`` for Multimodal model.

## Train and test
Run the ``train.sh`` to train and test the multimodal and accelerometer models. Log and weights would be saved under working directory. Use the following command to run the ``train.sh`` script.

```bash
./train.sh
```


