`timescale 1ns / 1ps

module tb_ln_topmodule;

    typedef logic signed [15:0] array_t [0:63];

    // 1. Signal Declaration
    reg           i_clk;
    reg           i_en;
    reg           i_rst;
    reg           i_valid;
    reg  [1023:0] i_data_flat;

    wire          o_valid;
    wire [1023:0] o_result_flat; 

    array_t       output_hw_array; 
    array_t       input_q[$]; 
    int           error_count = 0;

    // 2. DUT Instantiation (Top Module)
    ln_topmodule dut (
        .i_clk         (i_clk),
        .i_en          (i_en),
        .i_rst         (i_rst),
        .i_valid       (i_valid),
        .i_data_flat   (i_data_flat),
        .o_valid       (o_valid),
        .o_result_flat (o_result_flat)
    );

    // 3. Clock Generation
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // 4. Test Stimulus
    initial begin
        i_en = 1;
        i_rst = 1;
        i_valid = 0;
        i_data_flat = 0;

        #100;
        i_rst = 0;
        #20;

        $display("---------------------------------------------------");
        $display(" Layer Norm Top Module Simulation Start ");
        $display("---------------------------------------------------");

        repeat(50) begin
            send_packet_random();
        end
        
        @(posedge i_clk); 

        i_valid = 0;
        #1000;

        $display("---------------------------------------------------");
        if (error_count == 0) 
            $display(" [PASS] All tests passed within tolerance!");
        else 
            $display(" [FAIL] Total Errors: %d", error_count);
        $display("---------------------------------------------------");
        $finish;
    end

    // 5. Tasks
    task send_packet_random();
        array_t temp_arr;
        for(int k=0; k<64; k++) begin
            temp_arr[k] = $urandom_range(0, 200) - 100; 
        end
        drive_data(temp_arr);
    endtask

    task drive_data(input array_t data_in);
        input_q.push_back(data_in);
        @(posedge i_clk);
        i_valid <= 1;
        for (int k=0; k<64; k++) begin
            i_data_flat[16*k +: 16] <= data_in[k];
        end
    endtask

    always_comb begin
        for(int k=0; k<64; k++) begin
            output_hw_array[k] = o_result_flat[16*k +: 16];
        end
    end

    always @(posedge i_clk) begin
        if (!i_rst && o_valid) begin
            array_t original_input;
            
            if (input_q.size() > 0) begin
                original_input = input_q.pop_front();
            end else begin
                $display("[Error] o_valid asserted without input data!");
                $finish;
            end
            
            check_result(original_input, output_hw_array);
        end
    end

    function automatic void check_result(input array_t in_data, input array_t hw_out);
        real sum_x = 0;
        real sum_sq = 0;
        real mean, calc_var, inv_sqrt; 
        real expected_float;
        real actual_float;
        
        real scaling_factor = 91.2; 

        for(int k=0; k<64; k++) begin
            sum_x += in_data[k];
            sum_sq += real'(in_data[k]) * real'(in_data[k]);
        end
        
        mean = sum_x / 64.0;
        calc_var = (sum_sq / 64.0) - (mean * mean);
        if (calc_var < 0.001) calc_var = 0; 

        if (calc_var == 0) inv_sqrt = 0;
        else               inv_sqrt = 1.0 / $sqrt(calc_var); 

        for(int k=0; k<64; k++) begin
            expected_float = (real'(in_data[k]) - mean) * inv_sqrt * scaling_factor;
            actual_float = real'(hw_out[k]); 

            if (k == 0) begin
                $display("Time %t | In: %d | Mean: %0.1f | Var: %0.1f | Exp(x%0.1f): %0.1f | HW: %d", 
                         $time, in_data[k], mean, calc_var, scaling_factor, expected_float, hw_out[k]);
            end
        end
    endfunction

endmodule
