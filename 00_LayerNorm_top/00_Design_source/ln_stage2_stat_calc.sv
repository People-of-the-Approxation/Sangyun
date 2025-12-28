module ln_stage2_stat_calc (
    input  wire          i_clk,
    input  wire          i_rst,
    input  wire          i_en,
    
    input  wire          i_start,
    input  wire signed [30:0] i_sum,
    input  wire signed [50:0] i_sq_sum,

    output wire signed [31:0] o_mean, 
    output wire [15:0]        o_inv_sqrt,
    output wire               o_valid
);

    localparam PWL_LATENCY = 12;

    // [Step 1] 1차 곱셈 (IP Latency = 5)
    wire signed [42:0] w_mult_mean_raw;
    wire signed [62:0] w_mult_ex2_raw;
    
    mult_sum_const u_mult_mean (
        .CLK(i_clk),
        .CE(i_en),
        .A(i_sum),
        .B(12'd1365),
        .P(w_mult_mean_raw) 
    );

    mult_sq_sum_const u_mult_ex2 (
        .CLK(i_clk),
        .CE(i_en),
        .A(i_sq_sum),
        .B(12'd1365),
        .P(w_mult_ex2_raw)  
    );

    reg [4:0] r_val_d1; 
    always @(posedge i_clk) r_val_d1 <= {r_val_d1[3:0], i_start};

    // [Step 2] Shift & Register
    reg signed [31:0] r_mean_shifted;
    reg signed [31:0] r_ex2_shifted;
    reg               r_step2_valid;

    always @(posedge i_clk) begin
        if (r_val_d1[4]) begin 
            r_mean_shifted <= w_mult_mean_raw >>> 20;
            r_ex2_shifted  <= w_mult_ex2_raw  >>> 20;
            r_step2_valid  <= 1'b1;
        end else begin
            r_step2_valid  <= 1'b0;
        end
    end

    // [Step 3] 2차 곱셈 (Mean^2)

    wire signed [63:0] w_mean_sq_raw;
    
    mult_mean_sq u_mult_sq (
        .CLK(i_clk),
        .CE(i_en),
        .A(r_mean_shifted),
        .B(r_mean_shifted),
        .P(w_mean_sq_raw) 
    );

    // 데이터 Delay (IP Latency 맞춤 - 5클럭)
    reg signed [31:0] r_ex2_delay [0:4];
    reg signed [31:0] r_mean_delay [0:4];
    reg [4:0]         r_val_d2;
    integer d;

    always @(posedge i_clk) begin
        r_ex2_delay[0]  <= r_ex2_shifted;
        r_mean_delay[0] <= r_mean_shifted;
        r_val_d2        <= {r_val_d2[3:0], r_step2_valid};

        for(d=1; d<5; d=d+1) begin
            r_ex2_delay[d]  <= r_ex2_delay[d-1];
            r_mean_delay[d] <= r_mean_delay[d-1];
        end
    end

    // [Step 4] 뺄셈 및 Saturation (Variance)
    reg [15:0] var_for_pwl;
    reg        pwl_start_valid;
    reg signed [31:0] raw_var;

    always @(posedge i_clk) begin
        if (r_val_d2[4]) begin 
            // 분산 계산
            raw_var = r_ex2_delay[4] - w_mean_sq_raw[31:0];
            if (raw_var[31])            var_for_pwl <= 0;         
            else if (|raw_var[30:16])   var_for_pwl <= 16'hFFFF; 
            else                        var_for_pwl <= raw_var[15:0];
            
            pwl_start_valid <= 1'b1;
        end else begin
            pwl_start_valid <= 1'b0;
        end
    end

    // [Step 5] PWL (Inverse Sqrt) & Mean Synchronization
    
    // 1. PWL 모듈 인스턴스
    pwl_approx u_pwl (
        .i_clk(i_clk), .i_rst(i_rst),
        .i_en(1'b1),
        .i_valid(pwl_start_valid),
        .i_variance(var_for_pwl),
        .o_result(o_inv_sqrt),
        .o_valid(o_valid) 
    );

    reg signed [31:0] r_mean_sync [0:PWL_LATENCY-1];
    integer m;

    always @(posedge i_clk) begin
        if (i_en) begin
            r_mean_sync[0] <= r_mean_delay[4]; 
            for (m=1; m<PWL_LATENCY; m=m+1) begin
                r_mean_sync[m] <= r_mean_sync[m-1];
            end
        end
    end
    
    assign o_mean = r_mean_sync[PWL_LATENCY-1];

endmodule