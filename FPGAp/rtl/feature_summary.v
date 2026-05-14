// ============================================================
// feature_summary.v
//
// Project:
//   FPGA CNN Feature Extractor for fatigue monitoring
//
// Purpose:
//   This module receives pooled feature values one by one
//   and calculates summary features.
//
// For each feature map, it calculates:
//
//   1. sum
//   2. max
//   3. energy
//
// Input:
//   pooled feature values from MaxPool stage
//
// Feature value type:
//   unsigned 8-bit
//   range: 0 to 255
//
// Number of pooled values per kernel:
//   After Conv 3x3:
//      32x32 input -> 30x30 feature map
//
//   After MaxPool 2x2:
//      30x30 -> 15x15
//
//   So:
//      15 * 15 = 225 pooled values
//
// Output:
//   sum_out
//   max_out
//   energy_out
//   done
//
// ============================================================

module feature_summary #(
    parameter NUM_VALUES = 225
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,

    input  wire        data_valid,
    input  wire [7:0]  data_in,

    output reg  [19:0] sum_out,
    output reg  [7:0]  max_out,
    output reg  [31:0] energy_out,

    output reg         done
);

    // ------------------------------------------------------------
    // Internal counter
    // ------------------------------------------------------------
    // Counts how many pooled values have been received.
    //
    // NUM_VALUES = 225 for one 15x15 pooled feature map.
    // 8 bits are enough because 255 > 225.
    // ------------------------------------------------------------

    reg [7:0] count;

    // ------------------------------------------------------------
    // Square calculation for energy
    // ------------------------------------------------------------
    // energy = sum of data_in * data_in
    //
    // data_in max = 255
    // 255 * 255 = 65025
    //
    // 225 values:
    // 225 * 65025 = 14630625
    //
    // 32-bit is safe.
    // ------------------------------------------------------------

    wire [15:0] square_value;
    assign square_value = data_in * data_in;

    // ------------------------------------------------------------
    // Main sequential logic
    // ------------------------------------------------------------

    always @(posedge clk) begin
        if (rst) begin
            count      <= 8'd0;
            sum_out    <= 20'd0;
            max_out    <= 8'd0;
            energy_out <= 32'd0;
            done       <= 1'b0;
        end
        else begin
            // Default done is 0.
            // It becomes 1 for one clock when the last value is processed.
            done <= 1'b0;

            // Start clears the previous summary.
            // This prepares the module for a new feature map.
            if (start) begin
                count      <= 8'd0;
                sum_out    <= 20'd0;
                max_out    <= 8'd0;
                energy_out <= 32'd0;
                done       <= 1'b0;
            end
            else if (data_valid && !done) begin
                // Add current value to sum.
                sum_out <= sum_out + data_in;

                // Update max.
                if (data_in > max_out) begin
                    max_out <= data_in;
                end

                // Add square to energy.
                energy_out <= energy_out + square_value;

                // Count this value.
                if (count == NUM_VALUES - 1) begin
                    done <= 1'b1;
                    count <= 8'd0;
                end
                else begin
                    count <= count + 8'd1;
                end
            end
        end
    end

endmodule