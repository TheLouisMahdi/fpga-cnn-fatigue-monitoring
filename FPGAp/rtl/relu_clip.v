// ============================================================
// relu_clip.v
//
// Project:
//   FPGA CNN Feature Extractor for fatigue monitoring
//
// Purpose:
//   This module applies ReLU and clipping to a signed kernel output.
//
// Input:
//   signed 12-bit value from fixed_kernel_bank
//
// Output:
//   unsigned 8-bit feature value
//
// Operation:
//   if input < 0:
//       output = 0
//
//   else if input > 255:
//       output = 255
//
//   else:
//       output = input[7:0]
//
// Why?
//   Kernel outputs can be negative or larger than 255.
//   But feature maps are stored as 8-bit values.
// ============================================================

module relu_clip (
    input  wire signed [11:0] data_in,
    output reg  [7:0]         data_out
);

    always @(*) begin
        // Negative values become zero.
        // This is the ReLU part.
        if (data_in < 12'sd0) begin
            data_out = 8'd0;
        end

        // Values bigger than 255 are clipped to 255.
        // This prevents overflow when storing into 8-bit feature maps.
        else if (data_in > 12'sd255) begin
            data_out = 8'd255;
        end

        // Values between 0 and 255 pass through.
        else begin
            data_out = data_in[7:0];
        end
    end

endmodule