module pair_var (
    input  wire          i_clk,
    input  wire          i_en,
    input  wire          i_rst,
    input  wire          i_valid,
    input  wire [1023:0] i_data_flat, // 16bit * 64

    output wire          o_valid, 
    output wire [31:0]   o_variance,
    output wire [15:0]   o_mean
);

    reg signed [15:0] x [0:63];
    integer k;

    always @(*) begin
        for (k = 0; k < 64; k = k + 1) begin
            x[k] = i_data_flat[16*k +: 16];
        end
    end


    wire signed [31:0] x_sq [0:63];
    
    genvar g;
    generate
        for (g = 0; g < 64; g = g + 1) begin : gen_mult_inputs
            mult_gen_power_16 u_mult_input (
                .CLK (i_clk),
                .CE  (i_en),
                .A   (x[g]),    
                .B   (x[g]),    
                .P   (x_sq[g])  
            );
        end
    endgenerate

    // 3. Adder Tree
    reg signed [16:0] sum_stg1 [0:31];
    reg signed [17:0] sum_stg2 [0:15];
    reg signed [18:0] sum_stg3 [0:7];
    reg signed [19:0] sum_stg4 [0:3];
    reg signed [20:0] sum_stg5 [0:1];
    reg signed [21:0] sum_final;      

    reg signed [32:0] sq_stg1 [0:31];
    reg signed [33:0] sq_stg2 [0:15];
    reg signed [34:0] sq_stg3 [0:7];
    reg signed [35:0] sq_stg4 [0:3];
    reg signed [36:0] sq_stg5 [0:1];
    reg signed [37:0] sq_final;       

    integer i;
    always @(posedge i_clk) begin
        if (i_rst) begin
            sum_final <= 0;
            sq_final  <= 0;
        end else begin
            // Stage 1 (64 -> 32)
            for (i=0; i<32; i=i+1) begin
                sq_stg1[i]  <= x_sq[2*i] + x_sq[2*i+1];
                sum_stg1[i] <= x[2*i]    + x[2*i+1];
            end

            // Stage 2 (32 -> 16)
            for (i=0; i<16; i=i+1) begin
                sq_stg2[i]  <= sq_stg1[2*i] + sq_stg1[2*i+1];
                sum_stg2[i] <= sum_stg1[2*i] + sum_stg1[2*i+1];
            end

            // Stage 3 (16 -> 8)
            for (i=0; i<8; i=i+1) begin
                sq_stg3[i]  <= sq_stg2[2*i] + sq_stg2[2*i+1];
                sum_stg3[i] <= sum_stg2[2*i] + sum_stg2[2*i+1];
            end

            // Stage 4 (8 -> 4)
            for (i=0; i<4; i=i+1) begin
                sq_stg4[i]  <= sq_stg3[2*i] + sq_stg3[2*i+1];
                sum_stg4[i] <= sum_stg3[2*i] + sum_stg3[2*i+1];
            end

            // Stage 5 (4 -> 2)
            for (i=0; i<2; i=i+1) begin
                sq_stg5[i]  <= sq_stg4[2*i] + sq_stg4[2*i+1];
                sum_stg5[i] <= sum_stg4[2*i] + sum_stg4[2*i+1];
            end

            // Stage 6 (2 -> 1)
            sq_final  <= sq_stg5[0] + sq_stg5[1]; 
            sum_final <= sum_stg5[0] + sum_stg5[1]; 
        end
    end

    wire signed [43:0] sum_squared_raw; 
    wire signed [31:0] term_mean_sq;    
    wire signed [31:0] term_sq_mean;    
    wire signed [31:0] result_diff;     

    mult_gen_power_22 u_mult_mean (
        .CLK (i_clk),
        .CE  (i_en),
        .A   (sum_final), 
        .B   (sum_final), 
        .P   (sum_squared_raw)
    );

    assign term_sq_mean = sq_final[37:6];    
    assign term_mean_sq = sum_squared_raw[43:12]; 

    c_addsub_var u_sub (
        .CLK (i_clk),
        .CE  (i_en),
        .A   (term_sq_mean),
        .B   (term_mean_sq),
        .S   (result_diff)
    );

    // Variance Latency: Mult(5) + Tree(6) + Sub(3) = 14
    // Mean Latency: Tree(6) = 6

    reg signed [21:0] mean_delay_pipe [0:7]; // 8-stage delay line
    integer d;

    always @(posedge i_clk) begin
        if (i_rst) begin
            for(d=0; d<8; d=d+1) mean_delay_pipe[d] <= 0;
        end else if (i_en) begin
            mean_delay_pipe[0] <= sum_final;
            for(d=1; d<8; d=d+1) begin
                mean_delay_pipe[d] <= mean_delay_pipe[d-1];
            end
        end
    end

    // using the mean val about delay 8 clock data (14 - 6)
    wire signed [21:0] w_mean_delayed = mean_delay_pipe[7];

    reg [15:0] valid_sr; 
    localparam TOTAL_LATENCY = 14; 

    always @(posedge i_clk) begin
        if (i_rst) valid_sr <= 0;
        else       valid_sr <= {valid_sr[14:0], i_valid};
    end
    
    assign o_valid    = valid_sr[TOTAL_LATENCY-1];
    assign o_variance = result_diff;
    assign o_mean     = w_mean_delayed[21:6]; 

endmodule