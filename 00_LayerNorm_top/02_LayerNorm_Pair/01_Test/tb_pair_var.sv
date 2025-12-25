`timescale 1ns / 1ps

module tb_pair_var;
    // 1. Signal Declaration
    reg          i_clk;
    reg          i_en;
    reg          i_rst;
    reg          i_valid;
    reg [1023:0] i_data_flat;

    wire         o_valid;
    wire [31:0]  o_variance; // signed 32-bit

    logic signed [15:0] input_array [0:63];
    int                 expected_q[$];
    int                 error_count = 0;    // 에러 카운터

    // 2. DUT Instantiation
    pair_var dut (
        .i_clk       (i_clk),
        .i_en        (i_en),
        .i_rst       (i_rst),
        .i_valid     (i_valid),
        .i_data_flat (i_data_flat),
        .o_valid     (o_valid),
        .o_variance  (o_variance)
    );

    // 3. Clock Generation (100MHz)
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // 4. Golden Model
    function automatic int calculate_expected_variance(logic signed [15:0] inputs [0:63]);
        longint sum_x;
        longint sum_sq;
        longint mean_term;
        longint sq_mean_term;
        int     variance;
        
        sum_x  = 0;
        sum_sq = 0;

        // 1. Summation
        for (int i = 0; i < 64; i++) begin
            sum_x  += inputs[i];
            sum_sq += longint'(inputs[i]) * longint'(inputs[i]);
        end

        // 2. Hardware Bit Slicing Simulation 
        
        sq_mean_term = sum_sq >>> 6;            // E[X^2] = (SumSq >> 6)
        mean_term    = (sum_x * sum_x) >>> 12;  // (E[X])^2 = ((SumX)^2 >> 12)
        
        // 3. Subtraction
        variance = sq_mean_term - mean_term;
        
        return variance;
    endfunction

    // 5. Test Stimulus
    initial begin
        i_en        = 1; // Always Enable
        i_rst       = 1;
        i_valid     = 0;
        i_data_flat = 0;
        
        #100;
        i_rst = 0;
        #20;

        $display("---------------------------------------------------");
        $display(" Simulation Start ");
        $display("---------------------------------------------------");

        // Case 1: All Zeros (Variance should be 0)
        send_packet_uniform(0);

        // Case 2: All Same Value (Variance should be 0)
        send_packet_uniform(100);
        send_packet_uniform(-50);

        // Case 3: Alternating +10, -10 (Mean=0, Var=100)
        // logic: E[X]=0, E[X^2]=100 -> Var = 100 - 0 = 100
        send_packet_pattern_alt(10); 

        // Case 4: Random Burst (100 packets)
        repeat(100) begin
            send_packet_random();
        end

        // Wait for Pipeline Flush
        i_valid = 0;
        #500;

        $display("---------------------------------------------------");
        if (error_count == 0) 
            $display(" [PASS] All tests passed successfully!");
        else 
            $display(" [FAIL] Total Errors: %d", error_count);
        $display("---------------------------------------------------");
        $finish;
    end

    // 6. Tasks for Data Generation

    task send_packet_uniform(input logic signed [15:0] val);
        logic signed [15:0] temp_arr [0:63];
        for(int i=0; i<64; i++) temp_arr[i] = val;
        drive_data(temp_arr);
    endtask

    task send_packet_pattern_alt(input logic signed [15:0] val);
        logic signed [15:0] temp_arr [0:63];
        for(int i=0; i<64; i++) temp_arr[i] = (i%2==0) ? val : -val;
        drive_data(temp_arr);
    endtask

    task send_packet_random();
        logic signed [15:0] temp_arr [0:63];
        for(int i=0; i<64; i++) temp_arr[i] = $urandom_range(0, 65535) - 32768; // Signed random
        drive_data(temp_arr);
    endtask

    task drive_data(input logic signed [15:0] data_in [0:63]);
        int exp_val;
        
        // 1. Calculate Expected Result immediately
        exp_val = calculate_expected_variance(data_in);
        expected_q.push_back(exp_val);

        // 2. Drive RTL signals
        @(posedge i_clk);
        i_valid <= 1;
        // Flatten the array into 1024-bit vector
        for (int k=0; k<64; k++) begin
            i_data_flat[16*k +: 16] <= data_in[k];
        end
    endtask

    // =================================================================
    // 7. Output Monitor & Checker
    // =================================================================
    
    // Valid 신호가 High가 되면 Queue에서 예상값을 꺼내서 비교
    always @(posedge i_clk) begin
        // Reset 아닐 때 동작
        if (!i_rst) begin
            // Data Validity Toggle logic simulation (burst mode support)
            // 여기서는 i_valid를 burst로 주면 o_valid도 burst로 나옴
            
            if (o_valid) begin
                int expected;
                int actual;
                
                if (expected_q.size() == 0) begin
                    $display("[Error] o_valid asserted but no expected data in queue!");
                    error_count++;
                end else begin
                    expected = expected_q.pop_front(); // FIFO Pop
                    actual   = $signed(o_variance);

                    if (actual !== expected) begin
                        $display("[FAIL] Time: %t | Expected: %d | Actual: %d", 
                                 $time, expected, actual);
                        error_count++;
                    end else begin
                        // Success log (Optional: Comment out to reduce noise)
                        // $display("[OK] Time: %t | Variance: %d", $time, actual);
                    end
                end
            end
        end
    end

endmodule