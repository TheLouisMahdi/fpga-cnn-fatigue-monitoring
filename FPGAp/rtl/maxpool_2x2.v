// ============================================================
// maxpool_2x2.v
//
// Project:
//   FPGA CNN Feature Extractor for fatigue monitoring
//
// Purpose:
//   This module performs 2x2 Max Pooling.
//
// Input:
//   Four 8-bit feature values:
//
//      p00  p01
//      p10  p11
//
// Output:
//   The maximum value among the four inputs.
//
// Operation:
//   pool_out = max(p00, p01, p10, p11)
//
// Why MaxPool?
//   It reduces feature map size.
//   It keeps the strongest local feature.
//   It makes the feature extraction less sensitive to small shifts.
//
// Example:
//      12   90
//      40   30
//
//   Output:
//      90
// ============================================================

module maxpool_2x2 (
    input  wire [7:0] p00,
    input  wire [7:0] p01,
    input  wire [7:0] p10,
    input  wire [7:0] p11,

    output wire [7:0] pool_out
);

    wire [7:0] max_top;
    wire [7:0] max_bottom;

    // Maximum of the top row
    assign max_top = (p00 >= p01) ? p00 : p01;

    // Maximum of the bottom row
    assign max_bottom = (p10 >= p11) ? p10 : p11;

    // Maximum of both row maximums
    assign pool_out = (max_top >= max_bottom) ? max_top : max_bottom;

endmodule