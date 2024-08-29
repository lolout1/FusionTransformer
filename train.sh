#!/bin/bash
teacher_weights="withoutTB.pth"
teacher_dir="exps/utd/student/withoutTB"
student_dir="exps/berkley_woKD/inertial_transformer"
student_weights="ttfstudent.pth"
#teacher_dir="exps/UTD_wKD/ttf4"
result_file="result.txt"


# weights="berkley_best.pt"
# work_dir="exps/bmhad_woKD/late_fusion_epoch150_alldrop0.4"

# #Utd student without KD
# python3 main.py --config ./config/utd/student.yaml --model-saved-name $student_weights --work-dir $work_dir --device 7 --base-lr 2.5e-2 --include-val True
# python3 main.py --config ./config/utd/student.yaml --work-dir $work_dir  --weights "$work_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'


#Utd teacher 
python3 main.py --config ./config/utd/teacher.yaml --work-dir "$teacher_dir" --model-saved-name "$teacher_weights" --device 7  --base-lr 2.5e-3 --phase 'train' --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --work-dir "$teacher_dir" --weights "$teacher_dir/$teacher_weights" --device 7  --base-lr 2.5e-3 --phase 'test'

#utd student
#python3 main.py --config ./config/utd/student.yaml --work-dir "$student_dir"  --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/student.yaml --work-dir "$student_dir"  --weights "$student_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'


#distillation 
#python3 distiller.py --config ./config/utd/distill.yaml --work-dir $student_dir --device 7  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/utd/distill.yaml --work-dir "$student_dir" --weights "$student_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#berkley teacher 
#python3 main.py --config ./config/berkley/teacher.yaml --work-dir "$teacher_dir" --model-saved-name "$teacher_weights" --device 7  --base-lr 2.5e-3 --phase 'train' --include-val True
#python3 main.py --config ./config/berkley/teacher.yaml --work-dir "$teacher_dir" --weights "$teacher_dir/$teacher_weights" --device 7  --base-lr 2.5e-3 --phase 'test'

#berkley_student
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --weights "$student_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#utd student
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True


#czu 
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/czu/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#smartfallmm
#python3 main.py --config ./config/smartfallmm/teacher.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 6 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/graph.yaml --work-dir "$teacher_dir" --model-saved-name "$teacher_weights" --device 7  --base-lr 2.5e-3 --phase 'train' --include-val True