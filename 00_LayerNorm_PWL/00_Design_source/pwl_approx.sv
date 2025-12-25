// summary
// using ip catalog add & mult
// add (3 latency) : A 32bit, B 17bit, O 33bit, clk, CE
// mult(5 latency) : A 32bit, B 18bit, O 40ibt, clk, CE
// using unsigned Q5.11
// output s1.4.11
// total latency : 11 (each reg : 1 * 3EA / add : 3 / mult : 5)

module pwl_approx (
    input  wire        i_clk,
    input  wire        i_en,
    input  wire        i_rst,

    // Data input signals
    input  wire        i_valid,
    input  wire [15:0] i_variance,   // UQ5.11 Input

    // Data output signals
    output wire        o_valid,
    output wire [15:0] o_result      // S1.4.11 Output
);

    // Stage 0 & Stage 1 Registers
    reg [16:0] reg_stg0;
    reg [62:0] reg_stg1; // {valid, segment_idx, slope, base, mantissa}
    reg [16:0] reg_stg2; // Final Output Register

    // Internal signals (stage0 -> stage1)
    reg  [3:0]         segment_idx;
    reg  [15:0]        mantissa_extracted;
    reg  signed [23:0] slope_lut;
    reg  signed [17:0] base_lut;
    wire valid_stg0;

    // Internal signals (stage1 Unpacked)
    wire               stg1_valid;
    wire signed [23:0] stg1_slope;
    wire signed [17:0] stg1_base;
    wire [15:0]        stg1_mantissa;

    always @(posedge i_clk) begin
        if (i_rst) begin
            reg_stg0 <= 17'd0;
            reg_stg1 <= 63'd0;
        end
        else if (i_en) begin
            reg_stg0 <= {i_valid, i_variance};
            reg_stg1 <= {valid_stg0, segment_idx, slope_lut, base_lut, mantissa_extracted};
        end
    end

    assign valid_stg0 = reg_stg0[16];

    always @(*) begin
        casez (reg_stg0[15:0])
            16'b1???????????????: begin segment_idx = 4'd15; mantissa_extracted = { 1'b0, reg_stg0[14:0]}; end
            16'b01??????????????: begin segment_idx = 4'd14; mantissa_extracted = { 2'b0, reg_stg0[13:0]}; end
            16'b001?????????????: begin segment_idx = 4'd13; mantissa_extracted = { 3'b0, reg_stg0[12:0]}; end
            16'b0001????????????: begin segment_idx = 4'd12; mantissa_extracted = { 4'b0, reg_stg0[11:0]}; end
            16'b00001???????????: begin segment_idx = 4'd11; mantissa_extracted = { 5'b0, reg_stg0[10:0]}; end
            16'b000001??????????: begin segment_idx = 4'd10; mantissa_extracted = { 6'b0, reg_stg0[ 9:0]}; end
            16'b0000001?????????: begin segment_idx = 4'd9;  mantissa_extracted = { 7'b0, reg_stg0[ 8:0]}; end
            16'b00000001????????: begin segment_idx = 4'd8;  mantissa_extracted = { 8'b0, reg_stg0[ 7:0]}; end
            16'b000000001???????: begin segment_idx = 4'd7;  mantissa_extracted = { 9'b0, reg_stg0[ 6:0]}; end
            16'b0000000001??????: begin segment_idx = 4'd6;  mantissa_extracted = {10'b0, reg_stg0[ 5:0]}; end
            16'b00000000001?????: begin segment_idx = 4'd5;  mantissa_extracted = {11'b0, reg_stg0[ 4:0]}; end
            16'b000000000001????: begin segment_idx = 4'd4;  mantissa_extracted = {12'b0, reg_stg0[ 3:0]}; end
            16'b0000000000001???: begin segment_idx = 4'd3;  mantissa_extracted = {13'b0, reg_stg0[ 2:0]}; end
            16'b00000000000001??: begin segment_idx = 4'd2;  mantissa_extracted = {14'b0, reg_stg0[ 1:0]}; end
            16'b000000000000001?: begin segment_idx = 4'd1;  mantissa_extracted = {15'b0, reg_stg0[ 0:0]}; end
            default:              begin segment_idx = 4'd0;  mantissa_extracted = 16'd0;                 end
        endcase
    end

    always @(*) begin
        case (segment_idx)
            4'd15: begin slope_lut = -24'd1;       base_lut = 18'd497;   end
            4'd14: begin slope_lut = -24'd3;       base_lut = 18'd703;   end
            4'd13: begin slope_lut = -24'd9;       base_lut = 18'd994;   end
            4'd12: begin slope_lut = -24'd26;      base_lut = 18'd1406;  end
            4'd11: begin slope_lut = -24'd73;      base_lut = 18'd1988;  end
            4'd10: begin slope_lut = -24'd206;     base_lut = 18'd2812;  end
            4'd9:  begin slope_lut = -24'd583;     base_lut = 18'd3977;  end
            4'd8:  begin slope_lut = -24'd1648;    base_lut = 18'd5624;  end
            4'd7:  begin slope_lut = -24'd4662;    base_lut = 18'd7954;  end
            4'd6:  begin slope_lut = -24'd13185;   base_lut = 18'd11249; end
            4'd5:  begin slope_lut = -24'd37294;   base_lut = 18'd15908; end
            4'd4:  begin slope_lut = -24'd105483;  base_lut = 18'd22497; end
            4'd3:  begin slope_lut = -24'd298352;  base_lut = 18'd31816; end
            4'd2:  begin slope_lut = -24'd843868;  base_lut = 18'd44995; end
            4'd1:  begin slope_lut = -24'd2386819; base_lut = 18'd63632; end
            4'd0:  begin slope_lut = -24'd6750944; base_lut = 18'd89989; end
            default: begin slope_lut = 24'd0;      base_lut = 18'd0;     end
        endcase
    end

    assign stg1_valid       = reg_stg1[62];
    assign stg1_slope       = reg_stg1[57:34];
    assign stg1_base        = reg_stg1[33:16];
    assign stg1_mantissa    = reg_stg1[15:0];

    reg signed [17:0] base_delay_pipe [0:4]; // 5-stage shift register
    integer i;

    always @(posedge i_clk) begin
        if (i_rst) begin
            for(i=0; i<5; i=i+1) base_delay_pipe[i] <= 18'd0;
        end
        else if (i_en) begin
            base_delay_pipe[0] <= stg1_base;
            for(i=1; i<5; i=i+1) base_delay_pipe[i] <= base_delay_pipe[i-1];
        end
    end
    
    wire signed [17:0] w_base_delayed = base_delay_pipe[4];

    reg [7:0] valid_delay_pipe;

    always @(posedge i_clk) begin
        if (i_rst) valid_delay_pipe <= 8'd0;
        else if (i_en) begin
            valid_delay_pipe <= {valid_delay_pipe[6:0], stg1_valid};
        end
    end

    wire w_valid_final = valid_delay_pipe[7];


    // 3. IP Instantiation
    wire signed [39:0] w_mul_out;
    wire signed [32:0] w_add_out;

    // [Multiplier IP] Latency = 5
    // A: 24bit, B: 17bit
    mult_gen_pwl u_mult (
        .CLK (i_clk),
        .CE  (i_en),
        .A   (stg1_slope),
        .B   ({1'b0, stg1_mantissa}),
        .P   (w_mul_out)
    );

    // [Adder IP] Latency = 3
    // A: 32bit (Mult 결과 상위), B: 18bit (지연된 Base)
    c_addsub_pwl u_adder (
        .CLK (i_clk),
        .CE  (i_en),
        .A   (w_mul_out[39:8]),   // Shift Right 8
        .B   (w_base_delayed),
        .S   (w_add_out)
    );

    reg [15:0] result_saturated;
    reg signed [39:0] calc_final_extended;

    always @(*) begin
        calc_final_extended = $signed(w_add_out);

        // Saturation (S1.4.11)
        if (calc_final_extended > 40'sd32767)      result_saturated = 16'h7FFF;
        else if (calc_final_extended < -40'sd32768) result_saturated = 16'd0;
        else                                        result_saturated = calc_final_extended[15:0];
    end

    always @(posedge i_clk) begin
        if (i_rst) begin
            reg_stg2 <= 17'd0;
        end
        else if (i_en) begin
            reg_stg2 <= {w_valid_final, result_saturated};
        end
    end

    // Outputs
    assign o_valid  = reg_stg2[16];
    assign o_result = reg_stg2[15:0];

endmodule