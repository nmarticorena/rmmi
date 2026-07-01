folder=$1

mkdir -p "logs/random_reaching/$folder"

info_file="logs/random_reaching/$folder/info.txt"
# Add the date to the file
echo "Date: $(date)" > $info_file
echo "Task: Normal reaching with inequality constrains" >> $info_file


file="scripts/reaching/experiment_random.py"

envs=(table_new bookshelf_cage)

# sampled=("new_points_0" "new_points_1" "new_points_2" "new_points_3" "new_points_4" "points_1" "points_2" "points_3" "points_4" "points_5" "points_6" "points_7" "points_8" "points_10")
sampled=("new_points_10" "new_points_11")
active_col=(w_avg)

for r in ${sampled[@]}; do
  folder_r="$folder/$r"
  active_col=(w_avg)
  for e in ${envs[@]}; do
    for a in ${active_col[@]}; do
      python $file --exp-name "$folder_r" --env-name "$e" --config.collision-cost "$a"   --config.no-approx-jacobian  --robot $r &
    done
  done
  wait
  python3 scripts/reaching/get_results.py --exp-name "random_reaching/$folder_r" --envs ${envs[@]}
  wait
  python3 scripts/reaching/average_results.py --exp-name "random_reaching/$folder_r" --envs ${envs[@]} --file-name $folder
done

robots=(spheres)
for r in ${robots[@]}; do
  folder_r="$folder/$r"
  for e in ${envs[@]}; do
    for a in ${active_col[@]}; do
      python $file --exp-name "$folder_r" --env-name "$e" --config.collision-cost "$a"   --config.no-approx-jacobian --"$r" &
    done
  done
  wait
  python3 scripts/reaching/get_results.py --exp-name "random_reaching/$folder_r" --envs ${envs[@]}
  wait
  python3 scripts/reaching/average_results.py --exp-name "random_reaching/$folder_r" --envs ${envs[@]} --file-name $folder
done
