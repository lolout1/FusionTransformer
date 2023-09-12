#!/bin/bash

#Utd student without KD
#python3 main.py --config ./config/utd/student.yaml --work-dir exps/UTD_woKD/MM --device 0 --base-lr 2.5e-2 

#Utd teacher 
python3 main.py --config ./config/utd/teacher.yaml --work-dir exps/UTD_woKD/MM --device 0 --base-lr 2.5e-2
#python3 main.py --config ./config/utd/teacher.yaml --weights exps/UTD_woKD/MM/test.pt --device 0 --phase 'test'
