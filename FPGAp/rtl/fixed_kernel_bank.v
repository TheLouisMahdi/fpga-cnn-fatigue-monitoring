// ============================================================
// fixed_kernel_bank.v
//
// Project:
//   FPGA CNN Feature Extractor for fatigue monitoring
//
// Purpose:
//   This module receives one 3x3 pixel window and applies
//   several fixed 3x3 kernels in parallel.
//
// Why fixed kernels?
//   The goal is speed and low latency.
//   Fixed kernels do not need general multipliers.
//   They can be implemented using add, subtract, and shift.
//
// Input:
//   9 pixels from a 3x3 window
//
// Pixel type:
//   unsigned 8-bit grayscale
//   range: 0 to 255
//
// Output:
//   6 signed convolution results
//
// Kernels:
//   0. Vertical edge
//   1. Horizontal edge
//   2. Diagonal 45 degree
//   3. Diagonal 135 degree
//   4. Sharpen
//   5. Center
// ============================================================

module fixed_kernel_bank (
    input  wire [7:0] p00,
    input  wire [7:0] p01,
    input  wire [7:0] p02,

    input  wire [7:0] p10,
    input  wire [7:0] p11,
    input  wire [7:0] p12,

    input  wire [7:0] p20,
    input  wire [7:0] p21,
    input  wire [7:0] p22,

    output wire signed [11:0] k_vertical,
    output wire signed [11:0] k_horizontal,
    output wire signed [11:0] k_diag45,
    output wire signed [11:0] k_diag135,
    output wire signed [11:0] k_sharpen,
    output wire signed [11:0] k_center
);

    // ------------------------------------------------------------
    // Convert unsigned 8-bit pixels to signed 12-bit values.
    //
    // Why 12-bit?
    //   Pixel values are 0 to 255.
    //   Kernel outputs can become negative or bigger than 255.
    //   12-bit signed range is:
    //      -2048 to +2047
    //   This is enough for these fixed kernels.
    // ------------------------------------------------------------

    wire signed [11:0] s00 = {4'b0000, p00};
    wire signed [11:0] s01 = {4'b0000, p01};
    wire signed [11:0] s02 = {4'b0000, p02};

    wire signed [11:0] s10 = {4'b0000, p10};
    wire signed [11:0] s11 = {4'b0000, p11};
    wire signed [11:0] s12 = {4'b0000, p12};

    wire signed [11:0] s20 = {4'b0000, p20};
    wire signed [11:0] s21 = {4'b0000, p21};
    wire signed [11:0] s22 = {4'b0000, p22};

    // ------------------------------------------------------------
    // Kernel 0: Vertical edge
    //
    // Kernel:
    //   -1   0   1
    //   -1   0   1
    //   -1   0   1
    //
    // Meaning:
    //   Detects vertical brightness changes.
    //   Useful for side borders of eye, mouth, and face regions.
    //
    // Formula:
    //   right column minus left column
    // ------------------------------------------------------------

    assign k_vertical =
        (s02 + s12 + s22) -
        (s00 + s10 + s20);

    // ------------------------------------------------------------
    // Kernel 1: Horizontal edge
    //
    // Kernel:
    //   -1  -1  -1
    //    0   0   0
    //    1   1   1
    //
    // Meaning:
    //   Detects horizontal brightness changes.
    //   Very important for eyelids and lips.
    //
    // Formula:
    //   bottom row minus top row
    // ------------------------------------------------------------

    assign k_horizontal =
        (s20 + s21 + s22) -
        (s00 + s01 + s02);

    // ------------------------------------------------------------
    // Kernel 2: Diagonal 45 degree
    //
    // Kernel:
    //    0   1   1
    //   -1   0   1
    //   -1  -1   0
    //
    // Meaning:
    //   Detects diagonal edges in one direction.
    //   Useful for eye corners and mouth curves.
    // ------------------------------------------------------------

    assign k_diag45 =
        (s01 + s02 + s12) -
        (s10 + s20 + s21);

    // ------------------------------------------------------------
    // Kernel 3: Diagonal 135 degree
    //
    // Kernel:
    //    1   1   0
    //    1   0  -1
    //    0  -1  -1
    //
    // Meaning:
    //   Detects diagonal edges in the opposite direction.
    //   Complements k_diag45.
    // ------------------------------------------------------------

    assign k_diag135 =
        (s00 + s01 + s10) -
        (s12 + s21 + s22);

    // ------------------------------------------------------------
    // Kernel 4: Sharpen
    //
    // Kernel:
    //    0  -1   0
    //   -1   5  -1
    //    0  -1   0
    //
    // Meaning:
    //   Emphasizes the center pixel and subtracts neighbors.
    //   Useful for making eye and mouth boundaries stronger.
    //
    // Formula:
    //   5*p11 - p01 - p10 - p12 - p21
    //
    // Hardware optimization:
    //   5*p11 = 4*p11 + p11
    //   4*p11 = p11 << 2
    // ------------------------------------------------------------

    assign k_sharpen =
        ((s11 <<< 2) + s11) -
        (s01 + s10 + s12 + s21);

    // ------------------------------------------------------------
    // Kernel 5: Center
    //
    // Kernel:
    //    0   0   0
    //    0   1   0
    //    0   0   0
    //
    // Meaning:
    //   Passes the center pixel.
    //   Gives the FPGA a raw brightness feature.
    // ------------------------------------------------------------

    assign k_center = s11;

endmodule