file="scripts/reaching/post_process.py"
folder=$1

envs=(table_new)
# envs=(bookshelf bookshelf_2 table_free table table_cyl_small)
for e in ${envs[@]}; do
  python3 $file --envs $e --exp_name 'logs/'${folder}& 
done
wait
