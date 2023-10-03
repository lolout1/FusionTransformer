#!/bin/bash

#Utd student without KD
#python3 main.py --config ./config/utd/student.yaml --work-dir exps/UTD_woKD/MM --device 0 --base-lr 2.5e-2 

#Utd teacher 
python3 main.py --config ./config/utd/teacher.yaml --work-dir exps/UTD_woKD/MMNorm --device 0 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --weights exps/UTD_woKD/MMNorm/no_kd_aug.pt --device 0 --phase 'test'
#python3 main.py --config ./config/utd/teacher.yaml --weights exps/Sfmm_woKD/MM/no_kd_aug.pt --device 0 --phase 'test'
