`timescale 1ns/1ps

module tb_cnn_feature_extractor_all_top;

    reg clk;
    reg rst;
    reg start;

    wire busy;
    wire done;

    wire [575:0] left_features;
    wire [575:0] right_features;
    wire [575:0] mouth_features;

    cnn_feature_extractor_all_top dut (
        .clk(clk),
        .rst(rst),
        .start(start),

        .busy(busy),
        .done(done),

        .left_features(left_features),
        .right_features(right_features),
        .mouth_features(mouth_features)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    integer fout_left;
    integer fout_right;
    integer fout_mouth;
    integer i;

    initial begin
        rst = 1'b1;
        start = 1'b0;

        fout_left  = $fopen("mem/fpga_features_left_eye.txt", "w");
        fout_right = $fopen("mem/fpga_features_right_eye.txt", "w");
        fout_mouth = $fopen("mem/fpga_features_mouth.txt", "w");

        if (fout_left == 0) begin
            $display("ERROR: cannot open left output file");
            $stop;
        end

        if (fout_right == 0) begin
            $display("ERROR: cannot open right output file");
            $stop;
        end

        if (fout_mouth == 0) begin
            $display("ERROR: cannot open mouth output file");
            $stop;
        end

        #50;
        rst = 1'b0;

        #30;
        start = 1'b1;
        #10;
        start = 1'b0;

        $display("Simulation started for ALL TOP...");

        wait(done == 1'b1);

        $display("Simulation finished.");
        $display("============================================================");

        $display("LEFT EYE FEATURES");
        for (i = 0; i < 18; i = i + 1) begin
            $display("left feature[%0d] = %d", i, left_features[i*32 +: 32]);
            $fdisplay(fout_left, "%d", left_features[i*32 +: 32]);
        end

        $display("------------------------------------------------------------");

        $display("RIGHT EYE FEATURES");
        for (i = 0; i < 18; i = i + 1) begin
            $display("right feature[%0d] = %d", i, right_features[i*32 +: 32]);
            $fdisplay(fout_right, "%d", right_features[i*32 +: 32]);
        end

        $display("------------------------------------------------------------");

        $display("MOUTH FEATURES");
        for (i = 0; i < 18; i = i + 1) begin
            $display("mouth feature[%0d] = %d", i, mouth_features[i*32 +: 32]);
            $fdisplay(fout_mouth, "%d", mouth_features[i*32 +: 32]);
        end

        $display("============================================================");
        $display("Output files saved:");
        $display("mem/fpga_features_left_eye.txt");
        $display("mem/fpga_features_right_eye.txt");
        $display("mem/fpga_features_mouth.txt");

        $fclose(fout_left);
        $fclose(fout_right);
        $fclose(fout_mouth);

        #100;
        $stop;
    end

endmodule