exp_name=$1
n_trials=$2
file="scripts/real_robot/experiment_hardcoded.py"
logging_file="mm_neo/process_ros.py"
robot_name="real_robot"

env_name="new_table"

read -p "Start with active collision" a
read start
python $file --env-name $env_name --no-spheres --controller.collision-gain 0.75 --no-prepose --trials $n_trials --loop &
python $logging_file --robot_name $robot_name --exp-name ${env_name}_final_run_${exp_name}  &
wait


read -p "Start wo active collision" a
python $file --env-name ${env_name} --no-spheres --controller.collision-gain 0 --no-prepose --trials $n_trials --loop &
python $logging_file --robot_name $robot_name --exp-name ${env_name}_final_run_no_active &
wait

read -p "Start with active collision spheres" a
python $file --env-name ${env_name} --controller.collision-gain 0.75 --no-prepose --trials $n_trials --loop --spheres &
python $logging_file --robot_name "curobo_2" --exp-name ${env_name}_final_run_spheres_${exp_name}  --spheres &
wait


read -p "Start no collision" a
python $file --env-name ${env_name} --spheres --controller.collision-gain 0 --no-prepose --trials $n_trials --loop  --controller.no-collisions --controller.collision-cost "" &
python $logging_file --robot_name $robot_name --exp-name ${env_name}_final_run_no_col_${exp_name} &
wait
