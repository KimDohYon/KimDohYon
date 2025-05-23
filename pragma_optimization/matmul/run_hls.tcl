# HLS 프로젝트 설정
open_project matmul_proj
set_top top_matmul
add_files matmul.cpp

open_solution "solution1"
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# HLS 합성
csynth_design

# Vivado용 RTL export까지만 수행
export_design -format ip_catalog -rtl verilog -output /home/andrew/pragma_optimization_matmul/matmul_ip

# 결과 변수 초기화
set interval -1
set bram -1
set dsp -1
set ff -1
set lut -1

set rpt_file "matmul_proj/solution1/syn/report/csynth.rpt"
if {[file exists $rpt_file]} {
    set f [open $rpt_file r]
    set lines [split [read $f] "\n"]
    close $f

    foreach line $lines {
        if {[string match "*|+ top_matmul*" $line]} {
            set fields [split $line "|"]
            set fields_cleaned {}
            foreach f $fields {
                lappend fields_cleaned [string trim $f]
            }

            if {[llength $fields_cleaned] >= 15} {
                set interval        [lindex $fields_cleaned 7]
                set bram_raw        [lindex $fields_cleaned 10]
                set dsp_raw         [lindex $fields_cleaned 11]
                set ff_raw          [lindex $fields_cleaned 12]
                set lut_raw         [lindex $fields_cleaned 13]

                regexp {(\d+)} $bram_raw -> bram
                regexp {(\d+)} $dsp_raw  -> dsp
                regexp {(\d+)} $ff_raw   -> ff
                regexp {(\d+)} $lut_raw  -> lut
            }
            break
        }
    }
}

# DARTS 방식: pragma는 .cpp에 직접 삽입됨 → .py에서 current_pragma.txt로 따로 기록
set pragma_line "-1,-1,-1,-1,-1"
if {[file exists "current_pragma.txt"]} {
    set pf [open "current_pragma.txt" r]
    set pragma_line [gets $pf]
    close $pf
}

# 결과 저장
set fd [open "results.csv" "w"]
puts $fd "sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT"
puts $fd "$pragma_line,$interval,$bram,$dsp,$ff,$lut"
close $fd

exit

