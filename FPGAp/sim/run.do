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

transcript on

if {[file exists work]} {
    vdel -lib work -all
}

vlib work
vmap work work

# Compile RTL modules.
# This script is intended to be run from FPGAp/sim.
vlog ../rtl/*.v

# Compile testbench.
vlog ../tb/tb_cnn_feature_extractor_all_top.v

# Run simulation.
vsim work.tb_cnn_feature_extractor_all_top
add wave -r *
run -all
