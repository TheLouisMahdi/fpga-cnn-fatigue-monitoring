// ============================================================
// tb_cnn_feature_extractor_top.v
//
// Testbench for cnn_feature_extractor_top.v
//
// Purpose:
//   Simulate the complete FPGA CNN feature extractor.
//
// It tests one ROI file:
//
//   mem/left_eye.txt
//
// The testbench:
//   1. Generates clock
//   2. Applies reset
//   3. Sends start pulse
//   4. Waits for done
//   5. Prints 18 FPGA features
//   6. Saves features to output text file
// ============================================================

`timescale 1ns/1ps

module tb_cnn_feature_extractor_top;

    // ------------------------------------------------------------
    // Clock and control signals
    // ------------------------------------------------------------

    reg clk;
    reg rst;
    reg start;

    wire busy;
    wire done;

    // ------------------------------------------------------------
    // Output features
    // ------------------------------------------------------------

    wire [19:0] vertical_sum;
    wire [7:0]  vertical_max;
    wire [31:0] vertical_energy;

    wire [19:0] horizontal_sum;
    wire [7:0]  horizontal_max;
    wire [31:0] horizontal_energy;

    wire [19:0] diag45_sum;
    wire [7:0]  diag45_max;
    wire [31:0] diag45_energy;

    wire [19:0] diag135_sum;
    wire [7:0]  diag135_max;
    wire [31:0] diag135_energy;

    wire [19:0] sharpen_sum;
    wire [7:0]  sharpen_max;
    wire [31:0] sharpen_energy;

    wire [19:0] center_sum;
    wire [7:0]  center_max;
    wire [31:0] center_energy;

    // ------------------------------------------------------------
    // Instantiate DUT
    // ------------------------------------------------------------
    // If ModelSim is run from FPGApart folder:
    //   MEM_FILE = "mem/left_eye.txt"
    //
    // If ModelSim is run from sim folder:
    //   MEM_FILE = "../mem/left_eye.txt"
    // ------------------------------------------------------------

    cnn_feature_extractor_top #(
        .MEM_FILE("mem/left_eye.txt")
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),

        .busy(busy),
        .done(done),

        .vertical_sum(vertical_sum),
        .vertical_max(vertical_max),
        .vertical_energy(vertical_energy),

        .horizontal_sum(horizontal_sum),
        .horizontal_max(horizontal_max),
        .horizontal_energy(horizontal_energy),

        .diag45_sum(diag45_sum),
        .diag45_max(diag45_max),
        .diag45_energy(diag45_energy),

        .diag135_sum(diag135_sum),
        .diag135_max(diag135_max),
        .diag135_energy(diag135_energy),

        .sharpen_sum(sharpen_sum),
        .sharpen_max(sharpen_max),
        .sharpen_energy(sharpen_energy),

        .center_sum(center_sum),
        .center_max(center_max),
        .center_energy(center_energy)
    );

    // ------------------------------------------------------------
    // Clock generation
    // ------------------------------------------------------------
    // 10 ns period = 100 MHz simulation clock
    // ------------------------------------------------------------

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ------------------------------------------------------------
    // File handle
    // ------------------------------------------------------------

    integer fout;

    // ------------------------------------------------------------
    // Test sequence
    // ------------------------------------------------------------

    initial begin
        // Initial values
        rst   = 1'b1;
        start = 1'b0;

        // Open output file
        fout = $fopen("mem/fpga_features_left_eye.txt", "w");

        if (fout == 0) begin
            $display("ERROR: Could not open output file.");
            $stop;
        end

        // Reset
        #50;
        rst = 1'b0;

        // Small delay
        #30;

        // Start pulse
        start = 1'b1;
        #10;
        start = 1'b0;

        $display("Simulation started...");

        // Wait until done
        wait(done == 1'b1);

        $display("Simulation finished.");
        $display("--------------------------------------------");

        // Print outputs
        $display("VERTICAL   sum=%d max=%d energy=%d", vertical_sum, vertical_max, vertical_energy);
        $display("HORIZONTAL sum=%d max=%d energy=%d", horizontal_sum, horizontal_max, horizontal_energy);
        $display("DIAG45     sum=%d max=%d energy=%d", diag45_sum, diag45_max, diag45_energy);
        $display("DIAG135    sum=%d max=%d energy=%d", diag135_sum, diag135_max, diag135_energy);
        $display("SHARPEN    sum=%d max=%d energy=%d", sharpen_sum, sharpen_max, sharpen_energy);
        $display("CENTER     sum=%d max=%d energy=%d", center_sum, center_max, center_energy);

        // Save outputs to text file
        $fdisplay(fout, "%d", vertical_sum);
        $fdisplay(fout, "%d", vertical_max);
        $fdisplay(fout, "%d", vertical_energy);

        $fdisplay(fout, "%d", horizontal_sum);
        $fdisplay(fout, "%d", horizontal_max);
        $fdisplay(fout, "%d", horizontal_energy);

        $fdisplay(fout, "%d", diag45_sum);
        $fdisplay(fout, "%d", diag45_max);
        $fdisplay(fout, "%d", diag45_energy);

        $fdisplay(fout, "%d", diag135_sum);
        $fdisplay(fout, "%d", diag135_max);
        $fdisplay(fout, "%d", diag135_energy);

        $fdisplay(fout, "%d", sharpen_sum);
        $fdisplay(fout, "%d", sharpen_max);
        $fdisplay(fout, "%d", sharpen_energy);

        $fdisplay(fout, "%d", center_sum);
        $fdisplay(fout, "%d", center_max);
        $fdisplay(fout, "%d", center_energy);

        $fclose(fout);

        $display("--------------------------------------------");
        $display("Output saved to: mem/fpga_features_left_eye.txt");

        #100;
        $stop;
    end

endmodule