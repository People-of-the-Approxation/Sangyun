`timescale 1ns / 1ps

module ln_topmodule (
    input  wire          i_clk,
    input  wire          i_en,
    input  wire          i_rst,
    
    // Data Input
    input  wire          i_valid,
    input  wire [1023:0] i_data_flat, // 16bit * 64 Input Data

    // Data Output
    output wire          o_valid,
    output wire [1023:0] o_result_flat
);

    // 1. Internal Signals & Wires
    
    // pair_var Outputs
    wire        w_var_valid;
    wire [31:0] w_var_result_32; 
    wire [15:0] w_mean_16; 

    // pwl_approx Outputs
    wire        w_pwl_valid;
    wire [15:0] w_inv_sqrt;

    // Data Conversion for PWL
    reg  [15:0] r_var_input_16;  

    // 2. Instantiate Variance Calculation Module (pair_var)
    // Latency: 14 Cycles
    pair_var u_pair_var (
        .i_clk       (i_clk),
        .i_en        (i_en),
        .i_rst       (i_rst),
        .i_valid     (i_valid),
        .i_data_flat (i_data_flat),
        
        .o_valid     (w_var_valid),
        .o_variance  (w_var_result_32),
        .o_mean      (w_mean_16)
    );

    // 3. Data Conversion (Saturation 32bit -> 16bit)
    always @(*) begin
        if (w_var_result_32[31])           r_var_input_16 = 16'd0;     // 음수 처리
        else if (|w_var_result_32[30:16])  r_var_input_16 = 16'hFFFF;  // Overflow 처리
        else                               r_var_input_16 = w_var_result_32[15:0];
    end

    // 4. Instantiate PWL Approximation Module (pwl_approx)
    // Latency: 11 Cycles (Stage0(1)+Stage1(1)+Delay(8)+Stage2(1))
    // Total Latency from Input: 14 + 11 = 25 Cycles
    
    pwl_approx u_pwl_approx (
        .i_clk      (i_clk),
        .i_en       (i_en),
        .i_rst      (i_rst),
        .i_valid    (w_var_valid),    
        .i_variance (r_var_input_16), 
        .o_valid    (w_pwl_valid),
        .o_result   (w_inv_sqrt)
    );

    // 5. Delay Logic
    // 5-1. Input Data Delay (Latency 25)

    reg [1023:0] input_delay_pipe [0:24]; // Depth 25
    integer d;

    always @(posedge i_clk) begin
        if (i_en) begin
            input_delay_pipe[0] <= i_data_flat;
            for(d=1; d<25; d=d+1) begin
                input_delay_pipe[d] <= input_delay_pipe[d-1];
            end
        end
    end
    wire [1023:0] w_data_delayed = input_delay_pipe[24];


    // 5-2. Mean Value Delay (Latency 11)
    reg signed [15:0] mean_delay_pipe [0:10]; // Depth 11
    
    always @(posedge i_clk) begin
        if (i_en) begin
            mean_delay_pipe[0] <= w_mean_16;
            for(d=1; d<11; d=d+1) begin
                mean_delay_pipe[d] <= mean_delay_pipe[d-1];
            end
        end
    end
    wire signed [15:0] w_mean_delayed = mean_delay_pipe[10];

    // 6. Final Calculation (Normalization)
    reg signed [15:0] x_unpack [0:63];
    reg signed [31:0] calc_res [0:63];
    reg signed [15:0] final_res [0:63];
    
    genvar k;
    generate
        for (k=0; k<64; k=k+1) begin : gen_final_calc
            always @(*) x_unpack[k] = w_data_delayed[16*k +: 16];

            wire signed [16:0] diff;
            assign diff = x_unpack[k] - w_mean_delayed;
            
            always @(posedge i_clk) begin
                if (i_en) begin
                    calc_res[k] <= diff * $signed(w_inv_sqrt); 
                    final_res[k] <= calc_res[k][25:10];
                end
            end
            
            assign o_result_flat[16*k +: 16] = final_res[k];
        end
    endgenerate

    // 7. Output Valid
    // w_pwl_valid(25클럭) + Final Calc(1클럭) = 총 26 Latency
    
reg [1:0] valid_pipe;   // 2 bit shift reg

    always @(posedge i_clk) begin
        if (i_rst) begin
            valid_pipe <= 2'b0;
        end
        else if (i_en) begin
            valid_pipe <= {valid_pipe[0], w_pwl_valid}; 
        end
    end
    assign o_valid = valid_pipe[1];
endmodule