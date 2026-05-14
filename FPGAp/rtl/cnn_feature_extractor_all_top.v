// ============================================================
// cnn_feature_extractor_all_top.v
//
// Project:
//   FPGA CNN Feature Extractor for fatigue monitoring
//
// Purpose:
//   This top module processes all three ROI inputs:
//
//      1. left_eye.hex
//      2. right_eye.hex
//      3. mouth.hex
//
// It instantiates cnn_feature_extractor_top three times.
//
// Each ROI produces 18 features.
// Each feature is packed into a 32-bit slot.
//
// Output:
//   left_features  = 18 x 32-bit = 576 bits
//   right_features = 18 x 32-bit = 576 bits
//   mouth_features = 18 x 32-bit = 576 bits
//
// Feature order in each bus:
//
//   slot 0  = vertical_sum
//   slot 1  = vertical_max
//   slot 2  = vertical_energy
//
//   slot 3  = horizontal_sum
//   slot 4  = horizontal_max
//   slot 5  = horizontal_energy
//
//   slot 6  = diag45_sum
//   slot 7  = diag45_max
//   slot 8  = diag45_energy
//
//   slot 9  = diag135_sum
//   slot 10 = diag135_max
//   slot 11 = diag135_energy
//
//   slot 12 = sharpen_sum
//   slot 13 = sharpen_max
//   slot 14 = sharpen_energy
//
//   slot 15 = center_sum
//   slot 16 = center_max
//   slot 17 = center_energy
// ============================================================

module cnn_feature_extractor_all_top (
    input  wire clk,
    input  wire rst,
    input  wire start,

    output wire busy,
    output wire done,

    output wire [575:0] left_features,
    output wire [575:0] right_features,
    output wire [575:0] mouth_features
);

    // ============================================================
    // Busy / done signals for three ROI processors
    // ============================================================

    wire busy_left;
    wire busy_right;
    wire busy_mouth;

    wire done_left;
    wire done_right;
    wire done_mouth;

    assign busy = busy_left | busy_right | busy_mouth;
    assign done = done_left & done_right & done_mouth;

    // ============================================================
    // LEFT EYE feature wires
    // ============================================================

    wire [19:0] left_vertical_sum;
    wire [7:0]  left_vertical_max;
    wire [31:0] left_vertical_energy;

    wire [19:0] left_horizontal_sum;
    wire [7:0]  left_horizontal_max;
    wire [31:0] left_horizontal_energy;

    wire [19:0] left_diag45_sum;
    wire [7:0]  left_diag45_max;
    wire [31:0] left_diag45_energy;

    wire [19:0] left_diag135_sum;
    wire [7:0]  left_diag135_max;
    wire [31:0] left_diag135_energy;

    wire [19:0] left_sharpen_sum;
    wire [7:0]  left_sharpen_max;
    wire [31:0] left_sharpen_energy;

    wire [19:0] left_center_sum;
    wire [7:0]  left_center_max;
    wire [31:0] left_center_energy;

    // ============================================================
    // RIGHT EYE feature wires
    // ============================================================

    wire [19:0] right_vertical_sum;
    wire [7:0]  right_vertical_max;
    wire [31:0] right_vertical_energy;

    wire [19:0] right_horizontal_sum;
    wire [7:0]  right_horizontal_max;
    wire [31:0] right_horizontal_energy;

    wire [19:0] right_diag45_sum;
    wire [7:0]  right_diag45_max;
    wire [31:0] right_diag45_energy;

    wire [19:0] right_diag135_sum;
    wire [7:0]  right_diag135_max;
    wire [31:0] right_diag135_energy;

    wire [19:0] right_sharpen_sum;
    wire [7:0]  right_sharpen_max;
    wire [31:0] right_sharpen_energy;

    wire [19:0] right_center_sum;
    wire [7:0]  right_center_max;
    wire [31:0] right_center_energy;

    // ============================================================
    // MOUTH feature wires
    // ============================================================

    wire [19:0] mouth_vertical_sum;
    wire [7:0]  mouth_vertical_max;
    wire [31:0] mouth_vertical_energy;

    wire [19:0] mouth_horizontal_sum;
    wire [7:0]  mouth_horizontal_max;
    wire [31:0] mouth_horizontal_energy;

    wire [19:0] mouth_diag45_sum;
    wire [7:0]  mouth_diag45_max;
    wire [31:0] mouth_diag45_energy;

    wire [19:0] mouth_diag135_sum;
    wire [7:0]  mouth_diag135_max;
    wire [31:0] mouth_diag135_energy;

    wire [19:0] mouth_sharpen_sum;
    wire [7:0]  mouth_sharpen_max;
    wire [31:0] mouth_sharpen_energy;

    wire [19:0] mouth_center_sum;
    wire [7:0]  mouth_center_max;
    wire [31:0] mouth_center_energy;

    // ============================================================
    // Instance 1: LEFT EYE
    // ============================================================

    cnn_feature_extractor_top #(
        .MEM_FILE("mem/left_eye.hex")
    ) u_left_eye (
        .clk(clk),
        .rst(rst),
        .start(start),

        .busy(busy_left),
        .done(done_left),

        .vertical_sum(left_vertical_sum),
        .vertical_max(left_vertical_max),
        .vertical_energy(left_vertical_energy),

        .horizontal_sum(left_horizontal_sum),
        .horizontal_max(left_horizontal_max),
        .horizontal_energy(left_horizontal_energy),

        .diag45_sum(left_diag45_sum),
        .diag45_max(left_diag45_max),
        .diag45_energy(left_diag45_energy),

        .diag135_sum(left_diag135_sum),
        .diag135_max(left_diag135_max),
        .diag135_energy(left_diag135_energy),

        .sharpen_sum(left_sharpen_sum),
        .sharpen_max(left_sharpen_max),
        .sharpen_energy(left_sharpen_energy),

        .center_sum(left_center_sum),
        .center_max(left_center_max),
        .center_energy(left_center_energy)
    );

    // ============================================================
    // Instance 2: RIGHT EYE
    // ============================================================

    cnn_feature_extractor_top #(
        .MEM_FILE("mem/right_eye.hex")
    ) u_right_eye (
        .clk(clk),
        .rst(rst),
        .start(start),

        .busy(busy_right),
        .done(done_right),

        .vertical_sum(right_vertical_sum),
        .vertical_max(right_vertical_max),
        .vertical_energy(right_vertical_energy),

        .horizontal_sum(right_horizontal_sum),
        .horizontal_max(right_horizontal_max),
        .horizontal_energy(right_horizontal_energy),

        .diag45_sum(right_diag45_sum),
        .diag45_max(right_diag45_max),
        .diag45_energy(right_diag45_energy),

        .diag135_sum(right_diag135_sum),
        .diag135_max(right_diag135_max),
        .diag135_energy(right_diag135_energy),

        .sharpen_sum(right_sharpen_sum),
        .sharpen_max(right_sharpen_max),
        .sharpen_energy(right_sharpen_energy),

        .center_sum(right_center_sum),
        .center_max(right_center_max),
        .center_energy(right_center_energy)
    );

    // ============================================================
    // Instance 3: MOUTH
    // ============================================================

    cnn_feature_extractor_top #(
        .MEM_FILE("mem/mouth.hex")
    ) u_mouth (
        .clk(clk),
        .rst(rst),
        .start(start),

        .busy(busy_mouth),
        .done(done_mouth),

        .vertical_sum(mouth_vertical_sum),
        .vertical_max(mouth_vertical_max),
        .vertical_energy(mouth_vertical_energy),

        .horizontal_sum(mouth_horizontal_sum),
        .horizontal_max(mouth_horizontal_max),
        .horizontal_energy(mouth_horizontal_energy),

        .diag45_sum(mouth_diag45_sum),
        .diag45_max(mouth_diag45_max),
        .diag45_energy(mouth_diag45_energy),

        .diag135_sum(mouth_diag135_sum),
        .diag135_max(mouth_diag135_max),
        .diag135_energy(mouth_diag135_energy),

        .sharpen_sum(mouth_sharpen_sum),
        .sharpen_max(mouth_sharpen_max),
        .sharpen_energy(mouth_sharpen_energy),

        .center_sum(mouth_center_sum),
        .center_max(mouth_center_max),
        .center_energy(mouth_center_energy)
    );

    // ============================================================
    // Pack LEFT EYE features into 32-bit slots
    // ============================================================

    assign left_features[31:0]     = {12'd0, left_vertical_sum};
    assign left_features[63:32]    = {24'd0, left_vertical_max};
    assign left_features[95:64]    = left_vertical_energy;

    assign left_features[127:96]   = {12'd0, left_horizontal_sum};
    assign left_features[159:128]  = {24'd0, left_horizontal_max};
    assign left_features[191:160]  = left_horizontal_energy;

    assign left_features[223:192]  = {12'd0, left_diag45_sum};
    assign left_features[255:224]  = {24'd0, left_diag45_max};
    assign left_features[287:256]  = left_diag45_energy;

    assign left_features[319:288]  = {12'd0, left_diag135_sum};
    assign left_features[351:320]  = {24'd0, left_diag135_max};
    assign left_features[383:352]  = left_diag135_energy;

    assign left_features[415:384]  = {12'd0, left_sharpen_sum};
    assign left_features[447:416]  = {24'd0, left_sharpen_max};
    assign left_features[479:448]  = left_sharpen_energy;

    assign left_features[511:480]  = {12'd0, left_center_sum};
    assign left_features[543:512]  = {24'd0, left_center_max};
    assign left_features[575:544]  = left_center_energy;

    // ============================================================
    // Pack RIGHT EYE features into 32-bit slots
    // ============================================================

    assign right_features[31:0]     = {12'd0, right_vertical_sum};
    assign right_features[63:32]    = {24'd0, right_vertical_max};
    assign right_features[95:64]    = right_vertical_energy;

    assign right_features[127:96]   = {12'd0, right_horizontal_sum};
    assign right_features[159:128]  = {24'd0, right_horizontal_max};
    assign right_features[191:160]  = right_horizontal_energy;

    assign right_features[223:192]  = {12'd0, right_diag45_sum};
    assign right_features[255:224]  = {24'd0, right_diag45_max};
    assign right_features[287:256]  = right_diag45_energy;

    assign right_features[319:288]  = {12'd0, right_diag135_sum};
    assign right_features[351:320]  = {24'd0, right_diag135_max};
    assign right_features[383:352]  = right_diag135_energy;

    assign right_features[415:384]  = {12'd0, right_sharpen_sum};
    assign right_features[447:416]  = {24'd0, right_sharpen_max};
    assign right_features[479:448]  = right_sharpen_energy;

    assign right_features[511:480]  = {12'd0, right_center_sum};
    assign right_features[543:512]  = {24'd0, right_center_max};
    assign right_features[575:544]  = right_center_energy;

    // ============================================================
    // Pack MOUTH features into 32-bit slots
    // ============================================================

    assign mouth_features[31:0]     = {12'd0, mouth_vertical_sum};
    assign mouth_features[63:32]    = {24'd0, mouth_vertical_max};
    assign mouth_features[95:64]    = mouth_vertical_energy;

    assign mouth_features[127:96]   = {12'd0, mouth_horizontal_sum};
    assign mouth_features[159:128]  = {24'd0, mouth_horizontal_max};
    assign mouth_features[191:160]  = mouth_horizontal_energy;

    assign mouth_features[223:192]  = {12'd0, mouth_diag45_sum};
    assign mouth_features[255:224]  = {24'd0, mouth_diag45_max};
    assign mouth_features[287:256]  = mouth_diag45_energy;

    assign mouth_features[319:288]  = {12'd0, mouth_diag135_sum};
    assign mouth_features[351:320]  = {24'd0, mouth_diag135_max};
    assign mouth_features[383:352]  = mouth_diag135_energy;

    assign mouth_features[415:384]  = {12'd0, mouth_sharpen_sum};
    assign mouth_features[447:416]  = {24'd0, mouth_sharpen_max};
    assign mouth_features[479:448]  = mouth_sharpen_energy;

    assign mouth_features[511:480]  = {12'd0, mouth_center_sum};
    assign mouth_features[543:512]  = {24'd0, mouth_center_max};
    assign mouth_features[575:544]  = mouth_center_energy;

endmodule