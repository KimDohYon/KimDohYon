# =========================
# Vivado Project Setup
# =========================
create_project matmul_power ./vivado_proj -part xck26-sfvc784-2LV-c

set_property ip_repo_paths "/home/andrew/pragma_optimization_matmul/matmul_proj/solution1/impl/ip" [current_project]
update_ip_catalog

create_ip -name top_matmul_ffn -vendor xilinx.com -library hls -version 1.0 -module_name matmul_inst
set_property generate_synth_checkpoint false [get_files matmul_inst.xci]
generate_target all [get_ips *]

set_property source_mgmt_mode None [current_project]
update_compile_order -fileset sources_1
set_property top top_matmul_ffn [current_fileset]

synth_design -top top_matmul_ffn -part xck26-sfvc784-2LV-c
opt_design
place_design
route_design

set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]

# =========================
# Read environment safely
# =========================

proc get_env_or_default {var default} {
  if {[info exists ::env($var)] && [string is integer -strict $::env($var)]} {
    return $::env($var)
  } else {
    puts "Environment variable $var is missing or invalid. Using default $default"
    return $default
  }
}

set x_part  [get_env_or_default X_BUFF_PARTITION 1]
set xs_part [get_env_or_default XS_BUFF_PARTITION 1]
set w_part  [get_env_or_default W_BUFF_PARTITION 1]
set ws_part [get_env_or_default WS_BUFF_PARTITION 1]

# =========================
# Switching Activity
# =========================

create_clock -period 10 [get_ports ap_clk]
set_switching_activity -type clock -toggle_rate 100 -static_probability 0.5 [get_ports ap_clk]

foreach ctrl {ap_start ap_done ap_idle ap_ready ap_rst} {
  if {[llength [get_ports $ctrl]] > 0} {
    set_switching_activity -toggle_rate 10 -static_probability 0.1 [get_ports $ctrl]
  }
}

proc safe_toggle {factor} {
  if {![string is integer -strict $factor]} { return 10 }
  set raw [expr {5 * $factor}]
  return [expr {$raw > 80 ? 80 : $raw}]
}

proc safe_prob {toggle} {
  set min_prob [expr {double($toggle) / 200.0}]
  set max_prob [expr {1.0 - $min_prob}]
  set avg_prob [expr {($min_prob + $max_prob) / 2.0}]
  return [expr {$avg_prob > 0.01 ? $avg_prob : 0.05}]
}

set x_toggle   [safe_toggle $x_part]
set xs_toggle  [safe_toggle $xs_part]
set w_toggle   [safe_toggle $w_part]
set ws_toggle  [safe_toggle $ws_part]

set x_prob     [safe_prob $x_toggle]
set xs_prob    [safe_prob $xs_toggle]
set w_prob     [safe_prob $w_toggle]
set ws_prob    [safe_prob $ws_toggle]

# === 예시 포트 (IP 생성 후 실제 포트명 확인 필요) ===
foreach {port toggle prob} {
  xq_q0      $x_toggle   $x_prob
  xs_q0      $xs_toggle  $xs_prob
  wq_q0      $w_toggle   $w_prob
  ws_q0      $ws_toggle  $ws_prob
  o_ce0      $x_toggle   $x_prob
  o_we0      $x_toggle   $x_prob
} {
  if {[llength [get_ports $port]] > 0 && [string is integer -strict $toggle]} {
    set_switching_activity -toggle_rate $toggle -static_probability $prob [get_ports $port]
  } else {
    puts "Skipping $port due to invalid toggle ($toggle)"
  }
}

# =========================
# Power Report Generation
# =========================
puts "?? Trying to generate bitstream..."
if {[catch { write_bitstream -force ./vivado_proj/matmul_power.bit } result]} {
  puts "?? Bitstream generation failed: $result"
}

puts "? Running power report..."
if {[catch { report_power -file [file normalize "./vivado_proj/power_report.rpt"] } result]} {
  puts "? Power report failed: $result"
} else {
  puts "? Power report generated successfully."
}

puts "? Done."
catch { exit 0 }
