`timescale 1ns / 1ps

module tb_ln_stage2_stat_calc;

    // 1. 신호 선언
    reg          i_clk;
    reg          i_rst;
    reg          i_en;
    reg          i_start;
    reg signed [30:0] i_sum;
    reg signed [50:0] i_sq_sum;

    wire signed [31:0] o_mean;
    wire [15:0]        o_inv_sqrt;
    wire               o_valid;

    // 2. DUT 연결
    ln_stage2_stat_calc dut (
        .i_clk(i_clk), .i_rst(i_rst), .i_en(i_en),
        .i_start(i_start),
        .i_sum(i_sum), .i_sq_sum(i_sq_sum),
        .o_mean(o_mean), .o_inv_sqrt(o_inv_sqrt), .o_valid(o_valid)
    );

    // 3. 클럭 (100MHz)
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // 4. 테스트 시나리오
    integer k;

    initial begin
        // 초기화
        i_rst = 1; i_en = 0; i_start = 0; i_sum = 0; i_sq_sum = 0;
        
        #100;
        @(posedge i_clk);
        i_rst = 0; i_en = 1;

        $display("=== Streaming Test Start (Throughput = 1) ===");
        $display("Feeding 10 data items consecutively...");

        // =========================================================
        // [핵심] 매 사이클마다 새로운 데이터 밀어넣기 (Back-to-Back)
        // =========================================================
        for (k = 1; k <= 10; k = k + 1) begin
            @(posedge i_clk);
            i_start  <= 1; // 매번 Start 트리거
            
            // 데이터 값을 k에 따라 다르게 줌 (결과 구분을 위해)
            // 예: Sum = 6400, 12800, 19200 ...
            i_sum    <= k * 31'd6400;       
            i_sq_sum <= k * 51'd650000;     
        end

        // 10개 다 넣고 입력 끄기
        @(posedge i_clk);
        i_start <= 0;
        i_sum   <= 0;
        i_sq_sum <= 0;

        // 결과 다 나올 때까지 충분히 대기
        #500;
        $finish;
    end

    // 5. 결과 모니터링 (출력 카운트)
    integer out_cnt = 0;

    always @(posedge i_clk) begin
        if (o_valid) begin
            out_cnt = out_cnt + 1;
            $display("[Time %0t] Output #%2d Valid! | Mean: %d | InvSqrt: %d", 
                     $time, out_cnt, o_mean, o_inv_sqrt);
        end
    end

endmodule