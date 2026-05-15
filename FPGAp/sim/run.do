# ============================================================
# ModelSim simulation script
#
# This script verifies the FPGA accelerator side of the
# Python + FPGA fatigue monitoring project.
#
# Python generates:
#   ../mem/left_eye.hex
#   ../mem/right_eye.hex
#   ../mem/mouth.hex
#
# Verilog / ModelSim generates:
#   ../mem/fpga_features_left_eye.txt
#   ../mem/fpga_features_right_eye.txt
#   ../mem/fpga_features_mouth.txt
#
# Python golden_model.py compares:
#   fpga_features_*.txt
#   python_features_*.txt
#
# Goal:
#   Prove that the FPGA parallel feature extractor matches
#   the Python reference model.
# ============================================================

transcript file transcript.log

echo "============================================================"
echo "FPGA CNN Fatigue Monitoring"
echo "Python + FPGA co-processing verification"
echo "============================================================"

if {[file exists work]} {
    vdel -lib work -all
}

vlib work
vmap work work

echo "Compiling RTL..."

vlog ../rtl/input_ram.v
vlog ../rtl/fixed_kernel_bank.v
vlog ../rtl/relu_clip.v
vlog ../rtl/feature_summary.v
vlog ../rtl/maxpool_2x2.v
vlog ../rtl/cnn_feature_extractor_top.v
vlog ../rtl/cnn_feature_extractor_all_top.v

echo "Compiling testbench..."

vlog ../tb/tb_cnn_feature_extractor_all_top.v

echo "Starting simulation..."

vsim -voptargs=+acc work.tb_cnn_feature_extractor_all_top

add wave -divider "TB Control"
add wave sim:/tb_cnn_feature_extractor_all_top/clk
add wave sim:/tb_cnn_feature_extractor_all_top/rst
add wave sim:/tb_cnn_feature_extractor_all_top/start
add wave sim:/tb_cnn_feature_extractor_all_top/busy
add wave sim:/tb_cnn_feature_extractor_all_top/done

add wave -divider "DUT ROI Done"
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_left
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_right
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_mouth
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_left_latched
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_right_latched
add wave sim:/tb_cnn_feature_extractor_all_top/dut/done_mouth_latched

add wave -divider "DUT ROI Busy"
add wave sim:/tb_cnn_feature_extractor_all_top/dut/busy_left
add wave sim:/tb_cnn_feature_extractor_all_top/dut/busy_right
add wave sim:/tb_cnn_feature_extractor_all_top/dut/busy_mouth

add wave -divider "Feature Buses"
add wave -radix unsigned sim:/tb_cnn_feature_extractor_all_top/left_features
add wave -radix unsigned sim:/tb_cnn_feature_extractor_all_top/right_features
add wave -radix unsigned sim:/tb_cnn_feature_extractor_all_top/mouth_features

run -all

echo "============================================================"
echo "Simulation finished."
echo "Check generated files:"
echo "  ../mem/fpga_features_left_eye.txt"
echo "  ../mem/fpga_features_right_eye.txt"
echo "  ../mem/fpga_features_mouth.txt"
echo "Then run python/golden_model.py"
echo "============================================================"
