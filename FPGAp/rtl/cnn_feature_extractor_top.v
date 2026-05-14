// ============================================================
// cnn_feature_extractor_top.v
//
// Single ROI CNN Feature Extractor
//
// Input:
//   One 32x32 grayscale ROI image loaded by input_ram
//
// Output:
//   18 features:
//      6 kernels x 3 summary values
//
// Pipeline:
//   input_ram
//      |
//      v
//   3x3 window reader
//      |
//      v
//   fixed_kernel_bank
//      |
//      v
//   relu_clip
//      |
//      v
//   2x2 pooling
//      |
//      v
//   feature_summary
//
// Important fixes in this version:
//   1. RAM address is generated with correct 10-bit width.
//   2. RAM read timing is handled safely.
//      Address is set first.
//      Read enable is asserted after address is stable.
//      Data is captured only after ram_valid.
// ============================================================

module cnn_feature_extractor_top #(
    parameter MEM_FILE = "mem/left_eye.hex"
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    output reg  busy,
    output reg  done,

    output wire [19:0] vertical_sum,
    output wire [7:0]  vertical_max,
    output wire [31:0] vertical_energy,

    output wire [19:0] horizontal_sum,
    output wire [7:0]  horizontal_max,
    output wire [31:0] horizontal_energy,

    output wire [19:0] diag45_sum,
    output wire [7:0]  diag45_max,
    output wire [31:0] diag45_energy,

    output wire [19:0] diag135_sum,
    output wire [7:0]  diag135_max,
    output wire [31:0] diag135_energy,

    output wire [19:0] sharpen_sum,
    output wire [7:0]  sharpen_max,
    output wire [31:0] sharpen_energy,

    output wire [19:0] center_sum,
    output wire [7:0]  center_max,
    output wire [31:0] center_energy
);

    // ============================================================
    // Constants
    // ============================================================

    localparam IMG_SIZE      = 32;
    localparam POOL_SIZE     = 15;
    localparam TOTAL_POOLED  = 225;

    // ============================================================
    // FSM states
    // ============================================================

    localparam S_IDLE       = 4'd0;
    localparam S_SUM_START  = 4'd1;
    localparam S_PREP_POOL  = 4'd2;
    localparam S_PREP_WIN   = 4'd3;
    localparam S_SET_ADDR   = 4'd4;
    localparam S_LATCH_ADDR = 4'd5;
    localparam S_READ_REQ   = 4'd6;
    localparam S_CAPTURE    = 4'd7;
    localparam S_UPDATE_MAX = 4'd8;
    localparam S_SEND_SUM   = 4'd9;
    localparam S_NEXT_POOL  = 4'd10;
    localparam S_DONE       = 4'd11;

    reg [3:0] state;

    // ============================================================
    // Position counters
    // ============================================================

    reg [3:0] pool_row;
    reg [3:0] pool_col;

    reg [1:0] pool_win_id;
    reg [3:0] pixel_id;

    reg [5:0] win_base_row;
    reg [5:0] win_base_col;

    reg [5:0] read_row;
    reg [5:0] read_col;

    // ============================================================
    // Correct 10-bit address generation
    // ============================================================
    // Address mapping:
    //   addr = row * 32 + col
    //
    // row and col are 6-bit.
    // They must be expanded before shifting.
    // Otherwise Verilog may keep the shifted value too narrow.
    // ============================================================

    wire [9:0] read_addr_calc;

    assign read_addr_calc =
        ({4'b0000, read_row} << 5) + {4'b0000, read_col};

    // ============================================================
    // Input RAM
    // ============================================================

    reg        ram_rd_en;
    reg [9:0]  ram_addr;
    wire [7:0] ram_data;
    wire       ram_valid;

    input_ram #(
        .MEM_FILE(MEM_FILE)
    ) u_input_ram (
        .clk(clk),
        .rst(rst),
        .rd_en(ram_rd_en),
        .addr(ram_addr),
        .data_out(ram_data),
        .data_valid(ram_valid)
    );

    // ============================================================
    // 3x3 window registers
    // ============================================================

    reg [7:0] p00;
    reg [7:0] p01;
    reg [7:0] p02;

    reg [7:0] p10;
    reg [7:0] p11;
    reg [7:0] p12;

    reg [7:0] p20;
    reg [7:0] p21;
    reg [7:0] p22;

    // ============================================================
    // Fixed kernel bank
    // ============================================================

    wire signed [11:0] k_vertical;
    wire signed [11:0] k_horizontal;
    wire signed [11:0] k_diag45;
    wire signed [11:0] k_diag135;
    wire signed [11:0] k_sharpen;
    wire signed [11:0] k_center;

    fixed_kernel_bank u_fixed_kernel_bank (
        .p00(p00),
        .p01(p01),
        .p02(p02),

        .p10(p10),
        .p11(p11),
        .p12(p12),

        .p20(p20),
        .p21(p21),
        .p22(p22),

        .k_vertical(k_vertical),
        .k_horizontal(k_horizontal),
        .k_diag45(k_diag45),
        .k_diag135(k_diag135),
        .k_sharpen(k_sharpen),
        .k_center(k_center)
    );

    // ============================================================
    // ReLU and clipping for each kernel output
    // ============================================================

    wire [7:0] relu_vertical;
    wire [7:0] relu_horizontal;
    wire [7:0] relu_diag45;
    wire [7:0] relu_diag135;
    wire [7:0] relu_sharpen;
    wire [7:0] relu_center;

    relu_clip u_relu_vertical (
        .data_in(k_vertical),
        .data_out(relu_vertical)
    );

    relu_clip u_relu_horizontal (
        .data_in(k_horizontal),
        .data_out(relu_horizontal)
    );

    relu_clip u_relu_diag45 (
        .data_in(k_diag45),
        .data_out(relu_diag45)
    );

    relu_clip u_relu_diag135 (
        .data_in(k_diag135),
        .data_out(relu_diag135)
    );

    relu_clip u_relu_sharpen (
        .data_in(k_sharpen),
        .data_out(relu_sharpen)
    );

    relu_clip u_relu_center (
        .data_in(k_center),
        .data_out(relu_center)
    );

    // ============================================================
    // Pool max registers
    // ============================================================

    reg [7:0] pool_vertical;
    reg [7:0] pool_horizontal;
    reg [7:0] pool_diag45;
    reg [7:0] pool_diag135;
    reg [7:0] pool_sharpen;
    reg [7:0] pool_center;

    // ============================================================
    // Feature summary modules
    // ============================================================

    reg summary_start;
    reg summary_valid;

    wire summary_done_vertical;
    wire summary_done_horizontal;
    wire summary_done_diag45;
    wire summary_done_diag135;
    wire summary_done_sharpen;
    wire summary_done_center;

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_vertical (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_vertical),
        .sum_out(vertical_sum),
        .max_out(vertical_max),
        .energy_out(vertical_energy),
        .done(summary_done_vertical)
    );

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_horizontal (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_horizontal),
        .sum_out(horizontal_sum),
        .max_out(horizontal_max),
        .energy_out(horizontal_energy),
        .done(summary_done_horizontal)
    );

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_diag45 (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_diag45),
        .sum_out(diag45_sum),
        .max_out(diag45_max),
        .energy_out(diag45_energy),
        .done(summary_done_diag45)
    );

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_diag135 (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_diag135),
        .sum_out(diag135_sum),
        .max_out(diag135_max),
        .energy_out(diag135_energy),
        .done(summary_done_diag135)
    );

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_sharpen (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_sharpen),
        .sum_out(sharpen_sum),
        .max_out(sharpen_max),
        .energy_out(sharpen_energy),
        .done(summary_done_sharpen)
    );

    feature_summary #(
        .NUM_VALUES(TOTAL_POOLED)
    ) u_summary_center (
        .clk(clk),
        .rst(rst),
        .start(summary_start),
        .data_valid(summary_valid),
        .data_in(pool_center),
        .sum_out(center_sum),
        .max_out(center_max),
        .energy_out(center_energy),
        .done(summary_done_center)
    );

    // ============================================================
    // Capture RAM data into 3x3 window
    // ============================================================

    always @(posedge clk) begin
        if (rst) begin
            p00 <= 8'd0;
            p01 <= 8'd0;
            p02 <= 8'd0;

            p10 <= 8'd0;
            p11 <= 8'd0;
            p12 <= 8'd0;

            p20 <= 8'd0;
            p21 <= 8'd0;
            p22 <= 8'd0;
        end
        else begin
            if (state == S_CAPTURE && ram_valid) begin
                case (pixel_id)
                    4'd0: p00 <= ram_data;
                    4'd1: p01 <= ram_data;
                    4'd2: p02 <= ram_data;

                    4'd3: p10 <= ram_data;
                    4'd4: p11 <= ram_data;
                    4'd5: p12 <= ram_data;

                    4'd6: p20 <= ram_data;
                    4'd7: p21 <= ram_data;
                    4'd8: p22 <= ram_data;

                    default: begin
                        p00 <= p00;
                    end
                endcase
            end
        end
    end

    // ============================================================
    // Main FSM
    // ============================================================

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;

            busy <= 1'b0;
            done <= 1'b0;

            ram_rd_en <= 1'b0;
            ram_addr  <= 10'd0;

            summary_start <= 1'b0;
            summary_valid <= 1'b0;

            pool_row    <= 4'd0;
            pool_col    <= 4'd0;
            pool_win_id <= 2'd0;
            pixel_id    <= 4'd0;

            win_base_row <= 6'd0;
            win_base_col <= 6'd0;

            read_row <= 6'd0;
            read_col <= 6'd0;

            pool_vertical   <= 8'd0;
            pool_horizontal <= 8'd0;
            pool_diag45     <= 8'd0;
            pool_diag135    <= 8'd0;
            pool_sharpen    <= 8'd0;
            pool_center     <= 8'd0;
        end
        else begin
            ram_rd_en     <= 1'b0;
            summary_start <= 1'b0;
            summary_valid <= 1'b0;
            done          <= 1'b0;

            case (state)

                // ------------------------------------------------
                // Wait for start
                // ------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;

                    if (start) begin
                        busy <= 1'b1;

                        pool_row <= 4'd0;
                        pool_col <= 4'd0;

                        state <= S_SUM_START;
                    end
                end

                // ------------------------------------------------
                // Clear all summary modules
                // ------------------------------------------------
                S_SUM_START: begin
                    summary_start <= 1'b1;
                    state <= S_PREP_POOL;
                end

                // ------------------------------------------------
                // Start one 2x2 pooling region
                // ------------------------------------------------
                S_PREP_POOL: begin
                    pool_win_id <= 2'd0;

                    pool_vertical   <= 8'd0;
                    pool_horizontal <= 8'd0;
                    pool_diag45     <= 8'd0;
                    pool_diag135    <= 8'd0;
                    pool_sharpen    <= 8'd0;
                    pool_center     <= 8'd0;

                    state <= S_PREP_WIN;
                end

                // ------------------------------------------------
                // Select one convolution window inside 2x2 pool
                // ------------------------------------------------
                S_PREP_WIN: begin
                    pixel_id <= 4'd0;

                    case (pool_win_id)
                        2'd0: begin
                            win_base_row <= {2'b00, pool_row} << 1;
                            win_base_col <= {2'b00, pool_col} << 1;
                        end

                        2'd1: begin
                            win_base_row <= {2'b00, pool_row} << 1;
                            win_base_col <= ({2'b00, pool_col} << 1) + 6'd1;
                        end

                        2'd2: begin
                            win_base_row <= ({2'b00, pool_row} << 1) + 6'd1;
                            win_base_col <= {2'b00, pool_col} << 1;
                        end

                        2'd3: begin
                            win_base_row <= ({2'b00, pool_row} << 1) + 6'd1;
                            win_base_col <= ({2'b00, pool_col} << 1) + 6'd1;
                        end

                        default: begin
                            win_base_row <= 6'd0;
                            win_base_col <= 6'd0;
                        end
                    endcase

                    state <= S_SET_ADDR;
                end

                // ------------------------------------------------
                // Select row and column for one pixel in 3x3 window
                // ------------------------------------------------
                S_SET_ADDR: begin
                    case (pixel_id)
                        4'd0: begin
                            read_row <= win_base_row;
                            read_col <= win_base_col;
                        end

                        4'd1: begin
                            read_row <= win_base_row;
                            read_col <= win_base_col + 6'd1;
                        end

                        4'd2: begin
                            read_row <= win_base_row;
                            read_col <= win_base_col + 6'd2;
                        end

                        4'd3: begin
                            read_row <= win_base_row + 6'd1;
                            read_col <= win_base_col;
                        end

                        4'd4: begin
                            read_row <= win_base_row + 6'd1;
                            read_col <= win_base_col + 6'd1;
                        end

                        4'd5: begin
                            read_row <= win_base_row + 6'd1;
                            read_col <= win_base_col + 6'd2;
                        end

                        4'd6: begin
                            read_row <= win_base_row + 6'd2;
                            read_col <= win_base_col;
                        end

                        4'd7: begin
                            read_row <= win_base_row + 6'd2;
                            read_col <= win_base_col + 6'd1;
                        end

                        4'd8: begin
                            read_row <= win_base_row + 6'd2;
                            read_col <= win_base_col + 6'd2;
                        end

                        default: begin
                            read_row <= 6'd0;
                            read_col <= 6'd0;
                        end
                    endcase

                    state <= S_LATCH_ADDR;
                end

                // ------------------------------------------------
                // Latch RAM address after read_row/read_col are stable
                // ------------------------------------------------
                S_LATCH_ADDR: begin
                    ram_addr <= read_addr_calc;
                    state <= S_READ_REQ;
                end

                // ------------------------------------------------
                // Assert read enable after RAM address is stable
                // ------------------------------------------------
                S_READ_REQ: begin
                    ram_rd_en <= 1'b1;
                    state <= S_CAPTURE;
                end

                // ------------------------------------------------
                // Wait for RAM valid and capture data
                // ------------------------------------------------
                S_CAPTURE: begin
                    if (ram_valid) begin
                        if (pixel_id == 4'd8) begin
                            state <= S_UPDATE_MAX;
                        end
                        else begin
                            pixel_id <= pixel_id + 4'd1;
                            state <= S_SET_ADDR;
                        end
                    end
                end

                // ------------------------------------------------
                // Update 2x2 maxpool values
                // ------------------------------------------------
                S_UPDATE_MAX: begin
                    if (relu_vertical > pool_vertical)
                        pool_vertical <= relu_vertical;

                    if (relu_horizontal > pool_horizontal)
                        pool_horizontal <= relu_horizontal;

                    if (relu_diag45 > pool_diag45)
                        pool_diag45 <= relu_diag45;

                    if (relu_diag135 > pool_diag135)
                        pool_diag135 <= relu_diag135;

                    if (relu_sharpen > pool_sharpen)
                        pool_sharpen <= relu_sharpen;

                    if (relu_center > pool_center)
                        pool_center <= relu_center;

                    if (pool_win_id == 2'd3) begin
                        state <= S_SEND_SUM;
                    end
                    else begin
                        pool_win_id <= pool_win_id + 2'd1;
                        state <= S_PREP_WIN;
                    end
                end

                // ------------------------------------------------
                // Send one pooled value per kernel to summaries
                // ------------------------------------------------
                S_SEND_SUM: begin
                    summary_valid <= 1'b1;
                    state <= S_NEXT_POOL;
                end

                // ------------------------------------------------
                // Move to next pooled output location
                // ------------------------------------------------
                S_NEXT_POOL: begin
                    if (pool_row == POOL_SIZE - 1 && pool_col == POOL_SIZE - 1) begin
                        state <= S_DONE;
                    end
                    else begin
                        if (pool_col == POOL_SIZE - 1) begin
                            pool_col <= 4'd0;
                            pool_row <= pool_row + 4'd1;
                        end
                        else begin
                            pool_col <= pool_col + 4'd1;
                        end

                        state <= S_PREP_POOL;
                    end
                end

                // ------------------------------------------------
                // Processing finished
                // ------------------------------------------------
                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end

            endcase
        end
    end

endmodule