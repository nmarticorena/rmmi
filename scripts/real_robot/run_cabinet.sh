exp_name=$1
n_trials=$2
file="scripts/real_robot/experiment_hardcoded.py"
logging_file="mm_neo/process_ros.py"
robot_name="real_robot"

read -p "Start with active collision" a
read start
python $file --env-name cabinet_final --no-spheres --controller.collision-gain 1.0 --prepose --trials $n_trials --no-loop &
python $logging_file --robot_name $robot_name --exp-name cabinet_points_${exp_name}  &
wait

read -p "Start wo active collision" a
python $file --env-name cabinet_final --no-spheres --controller.collision-gain 0 --prepose --trials $n_trials --no-loop &
python $logging_file --robot_name $robot_name --exp-name cabinet_points_no_active_${exp_name} &
wait

read -p "Start with active collision spheres" a
python $file --env-name cabinet_final --controller.collision-gain 1.0 --prepose --trials $n_trials --no-loop --spheres &
python $logging_file --robot_name "curobo_2" --exp-name cabinet_spheres_${exp_name}  --spheres &
wait


read -p "Start wo active collision spheres" a
python $file --env-name cabinet_final --controller.collision-gain 0 --prepose --trials $n_trials --no-loop --spheres &
python $logging_file --robot_name "curobo_2" --exp-name cabinet_spheres_no_active_${exp_name}  --spheres &
wait

read -p "Start no collision" a
python $file --env-name cabinet_final --spheres --controller.collision-gain 0 --prepose --trials $n_trials --no-loop  --controller.no-collisions --controller.collision-cost "" &
python $logging_file --robot_name $robot_name --exp-name cabinet_no_col_${exp_name} &
wait
