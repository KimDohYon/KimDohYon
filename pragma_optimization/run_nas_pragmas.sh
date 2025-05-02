

#!/bin/bash

if [[ ! -f final_results.csv ]]; then
  echo "sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT,Power_W,score,epsilon,loss" > final_results.csv
fi

if [[ ! -f results.csv ]]; then
  echo "sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT,Power_W,score,epsilon,loss" > results.csv
fi

DONE=$(($(wc -l < final_results.csv) - 1))
NUM_EXPERIMENTS=1000

for ((i=DONE+1; i<=NUM_EXPERIMENTS; i++)); do
  PRAGMAS=$(python3 nas_model_dqn.py 2>/dev/null)
  IFS=',' read sum norm xbuff wbuff obuff <<< "$PRAGMAS"

  echo "[$i/$NUM_EXPERIMENTS] Running with: $PRAGMAS"

  export SUM_UNROLL=$sum
  export NORM_UNROLL=$norm
  export X_BUFF_PARTITION=$xbuff
  export WEIGHT_BUFF_PARTITION=$wbuff
  export OUT_BUFF_PARTITION=$obuff

  rm -rf rmsnorm_proj results.csv vivado_proj vitis_log_$i.txt vivado_log_$i.txt

  timeout 1200 vitis_hls -f run_hls.tcl > vitis_log_$i.txt 2>&1
  if [[ $? -ne 0 ]]; then
    echo "$sum,$norm,$xbuff,$wbuff,$obuff,-1,-1,-1,-1,-1,-1,0,0,0" >> final_results.csv
    continue
  fi

  timeout 1200 vivado -mode batch -source run_vivado_power.tcl > vivado_log_$i.txt 2>&1
  if [[ $? -ne 0 ]]; then
    echo "$sum,$norm,$xbuff,$wbuff,$obuff,-1,-1,-1,-1,-1,-1,0,0,0" >> final_results.csv
    continue
  fi

  if [[ -s vivado_proj/power_report.rpt && -f results.csv ]]; then
    echo "Completed experiment $i successfully"
  else
    echo "$sum,$norm,$xbuff,$wbuff,$obuff,-1,-1,-1,-1,-1,-1,0,0,0" >> final_results.csv
  fi
done
