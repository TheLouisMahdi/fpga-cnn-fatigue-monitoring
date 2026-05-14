// ============================================================
// input_ram.v
//
// Synthesizable input RAM for FPGA CNN Feature Extractor
//
// This version is suitable for Quartus synthesis.
//
// Input file format:
//   HEX text file
//   One 8-bit pixel per line
//
// Example:
//   A0
//   A0
//   A1
//   9F
//
// Image:
//   32 x 32 = 1024 pixels
//
// Address:
//   addr = row * 32 + col
// ============================================================

module input_ram #(
    parameter MEM_FILE = "mem/left_eye.hex"
)(
    input  wire        clk,
    input  wire        rst,

    input  wire        rd_en,
    input  wire [9:0]  addr,

    output reg  [7:0]  data_out,
    output reg         data_valid
);

    // 1024 locations, each 8-bit
    (* ramstyle = "M9K" *) reg [7:0] mem [0:1023];

    integer i;

    initial begin
        for (i = 0; i < 1024; i = i + 1) begin
            mem[i] = 8'd0;
        end

        // HEX memory initialization
        // This is supported for FPGA memory initialization.
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        if (rst) begin
            data_out   <= 8'd0;
            data_valid <= 1'b0;
        end
        else begin
            if (rd_en) begin
                data_out <= mem[addr];
            end

            data_valid <= rd_en;
        end
    end

endmodule