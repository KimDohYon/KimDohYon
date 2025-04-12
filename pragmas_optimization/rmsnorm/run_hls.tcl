open_project rmsnorm_proj
set_top top_rmsnorm
add_files rmsnorm.cpp

open_solution "solution1"
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# HLS 합성
csynth_design

# Vivado용 RTL export까지만 수행
# 수정할 코드 ?
export_design -format ip_catalog -rtl verilog -output /home/andrew/rmsnorm_NAS/rmsnorm_ip



# HLS 합성 결과 초기화
set interval -1
set bram -1
set dsp -1
set ff -1
set lut -1

set rpt_file "rmsnorm_proj/solution1/syn/report/csynth.rpt"

if {[file exists $rpt_file]} {
    set f [open $rpt_file r]
    set lines [split [read $f] "\n"]
    close $f

    foreach line $lines {
        if {[string match "*|+ rmsnorm*" $line]} {
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

                regexp {(\d+)} $bram_raw -> bram
                regexp {(\d+)} $dsp_raw  -> dsp
                regexp {(\d+)} $ff_raw   -> ff
            }
            break
        }
    }
}

# 결과 저장 (전력 정보는 Vivado 실행 후 별도로 추가할 것)
set fd [open "results.csv" "w"]
puts $fd "sum_unroll,norm_unroll,x_buff_partition,weight_buff_partition,out_buff_partition,interval,BRAM,DSP,FF,LUT"
puts $fd "$::env(SUM_UNROLL),$::env(NORM_UNROLL),$::env(X_BUFF_PARTITION),$::env(WEIGHT_BUFF_PARTITION),$::env(OUT_BUFF_PARTITION),$interval,$bram,$dsp,$ff,$lut"
close $fd

exit
