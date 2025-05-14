#!/bin/bash

# 파일 헤더 수정
if [[ ! -f final_results.csv ]]; then
  echo "x_buff_partition,xs_buff_partition,w_buff_partition,ws_buff_partition,interval,BRAM,DSP,FF,LUT,Power_W,score,alpha,loss" > final_results.csv
fi

if [[ ! -f results.csv ]]; then
  echo "x_buff_partition,xs_buff_partition,w_buff_partition,ws_buff_partition,interval,BRAM,DSP,FF,LUT" > results.csv
fi

DONE=$(($(wc -l < final_results.csv) - 1))
NUM_EXPERIMENTS=1000

for ((i=DONE+1; i<=NUM_EXPERIMENTS; i++)); do
  PRAGMAS=$(python3 nas_model_dqn.py 2>/dev/null)
  IFS=',' read xbuff xsbuff wbuff wsbuff <<< "$PRAGMAS"

  echo "[$i/$NUM_EXPERIMENTS] Running with: $PRAGMAS"

  export X_BUFF_PARTITION=$xbuff
  export XS_BUFF_PARTITION=$xsbuff
  export W_BUFF_PARTITION=$wbuff
  export WS_BUFF_PARTITION=$wsbuff

  rm -rf matmul_proj vivado_proj vitis_log_$i.txt vivado_log_$i.txt

  timeout 1200 vitis_hls -f run_hls.tcl > vitis_log_$i.txt 2>&1
  if [[ $? -ne 0 ]]; then
    echo "HLS failed at iteration $i, skipping..."
    continue
  fi

  timeout 1200 vivado -mode batch -source run_vivado_power.tcl > vivado_log_$i.txt 2>&1
  if [[ $? -ne 0 ]]; then
    echo "Vivado failed at iteration $i, skipping..."
    continue
  fi

  if [[ -s vivado_proj/power_report.rpt && -f results.csv ]]; then
    echo "Completed experiment $i successfully"
  else
    echo "Post-processing failed at iteration $i"
  fi

done

