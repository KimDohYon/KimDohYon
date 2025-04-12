

# -*- coding: utf-8 -*-
import os
import sys
import re

def parse_power_report(filepath="vivado_proj/power_report.rpt"):
    try:
        with open(filepath, "r") as f:
            for line in f:
                if "Total On-Chip Power" in line:
                    match = re.search(r"([\d.]+)", line)
                    if match:
                        return float(match.group(1))
    except FileNotFoundError:
        return -1.0
    return -1.0

def parse_results(filepath="results.csv"):
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("sum_")]
            fields = lines[-1].split(",")
            interval = float(fields[5])
            return fields[:10], interval
    except Exception:
        return ["-1"] * 10, -1.0

def load_baseline(filepath="baseline.txt"):
    try:
        with open(filepath, "r") as f:
            return tuple(map(float, f.read().strip().split(",")))
    except:
        return -1.0, -1.0

def evaluate_score(interval, power, base_interval, base_power, alpha=0.5, beta=0.5):
    if interval < 0 or power < 0 or base_interval <= 0 or base_power <= 0:
        return -1.0
    norm_p = power / base_power
    norm_i = interval / base_interval
    return round(1.0 / (alpha * norm_p + beta * norm_i + 1e-8), 6)

def update_final_results(results_fields, power, score, epsilon=None, loss=None, filepath="final_results.csv"):
    if len(results_fields) < 10 or results_fields[0] == "-1":
        return
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT,Power_W,score,epsilon,loss\n")
    line = f"{','.join(results_fields)},{power:.6f},{score:.6f},{epsilon:.6f},{loss:.6f}"
    with open(filepath, "a") as f:
        f.write(line + "\n")

def append_results_with_metrics(pragmas, interval, results_fields, power, score, epsilon, loss, filepath="results.csv"):
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT,Power_W,score,epsilon,loss\n")
    line = f"{','.join(map(str, pragmas))},{interval},{','.join(results_fields[1:])},{power:.6f},{score:.6f},{epsilon:.6f},{loss:.6f}"
    with open(filepath, "a") as f:
        f.write(line + "\n")
