# Fall Detection KD Multimodal 
## Getting started 
- Create an conda environment and use the requirements.txt to install all the neccasary files.
- For fall detection task choose *young* participant and for har detection task choose *old* . `num_classes` variable in the config file also needs to be changed from 2 to 8 for HAR Detection.
```bash
python3.8 -m venv smfall
```
```bash
source smfall/bin/activate
```
```bash
cd Fall_Detection_KD_Multimodal
```
```bash
pip install -r requirements.txt
```

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Get the Dataset

- Find all the datasets and npz files [here](https://txst-my.sharepoint.com/:f:/g/personal/bgu9_txstate_edu/EgHHgZoUISxDoY5uBHCwfOQBhQj89or79AC2A5Z98vToSA?e=98nB7i). 
- Download the data and put it under the root directory. 
- The datasets can processed from the scartch using the Processing_data.ipynb file.

## Choose and configure models
- Model configuration for Accelerometer model is kept under ``config/utd/student.yaml`` and in ``config/utd/teacher.yaml`` for  UTD Multimodal model.

- Model Configuration or Skeleton model is kept under ``config/smartfallmm/teacher.yaml`` for SmartFallMM dataset.

## Train and test
Give execution access to ``train.sh`` with 
```bash
chmod +x ./train.sh
```
Run the ``train.sh`` to train and test the multimodal and accelerometer models. Log and weights would be saved under working directory. Use the following command to run the ``train.sh`` script.

```bash
./train.sh
```


