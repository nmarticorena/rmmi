
version=$1
exp_name=$2


file="scripts/real_robot/experiment_hardcoded.py"
logging_file="mm_neo/process_ros.py"
robot_name="real_robot"
echo $version
read -p "Confirm $version " a

read -p "Start with active collision" a
read start
python $file --env-name s12_table_var_${version} --no-spheres --controller.collision-gain 0.75 --prepose --trials 1 --no-loop &
python $logging_file --robot_name $robot_name --exp-name s12_table_v2_var_${version}_${exp_name}  &
wait


# read -p "Start wo active collision" a
# python $file --env-name s12_table_var_${version} --no-spheres --controller.collision-gain 0 --prepose --trials 1 --no-loop &
# python $logging_file --robot_name $robot_name --exp-name s12_table_v2_var_${version}_no_active &

wait

read -p "Start with active collision spheres" a
python $file --env-name s12_table_var_${version} --controller.collision-gain 0.75 --prepose --trials 1 --no-loop --spheres &
python $logging_file --robot_name "curobo_4" --exp-name s12_table_v2_var_${version}_spheres${exp_name}  --spheres &
wait

