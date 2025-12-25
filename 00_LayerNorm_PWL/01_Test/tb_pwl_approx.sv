`timescale 1ns / 1ps

module tb_pwl_approx_burst;

    // =================================================================
    // 1. Signal Declaration
    // =================================================================
    reg         i_clk;
    reg         i_en;
    reg         i_rst;
    reg         i_valid;
    reg  [15:0] i_variance; // UQ5.11

    wire        o_valid;
    wire [15:0] o_result;   // S1.4.11

    // Debugging Variables (Real number for visualization)
    real        real_input;
    real        real_output;

    // =================================================================
    // 2. DUT Instantiation
    // =================================================================
    pwl_approx dut (
        .i_clk      (i_clk),
        .i_en       (i_en),
        .i_rst      (i_rst),
        .i_valid    (i_valid),
        .i_variance (i_variance),
        .o_valid    (o_valid),
        .o_result   (o_result)
    );

    // =================================================================
    // 3. Clock Generation (100MHz)
    // =================================================================
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // Real Number Conversion
    always @(*) real_input  = i_variance / 2048.0;
    always @(*) real_output = o_result / 2048.0;

    // =================================================================
    // 4. Test Stimulus (Main Sequence)
    // =================================================================
    integer k;

    initial begin
        // --- Initialization ---
        i_en       = 0;
        i_rst      = 1;
        i_valid    = 0;
        i_variance = 0;

        // Reset Pulse
        #100;
        i_rst = 0;
        i_en  = 1; // Pipeline Enable
        #20;

        $display("----------------------------------------------------------------");
        $display(" Simulation Start: 30 Burst -> Gap -> 20 Burst");
        $display("----------------------------------------------------------------");

        // ---------------------------------------------------------
        // Phase 1: 30 Continuous Inputs (0.5 ~ 15.0)
        // ---------------------------------------------------------
        $display("[Time %t] Phase 1: Sending 30 Data items...", $time);
        
        for (k = 1; k <= 30; k = k + 1) begin
            send_data(k * 0.5); // 0.5, 1.0, 1.5 ... 15.0
        end

        // ---------------------------------------------------------
        // Phase 2: Gap (Idle for 20 cycles)
        // ---------------------------------------------------------
        @(posedge i_clk);
        i_valid <= 0; // Stop Valid
        i_variance <= 0;
        
        $display("[Time %t] Phase 2: Gap (Idle for 20 cycles)...", $time);
        repeat(20) @(posedge i_clk); // Wait 20 clocks

        // ---------------------------------------------------------
        // Phase 3: 20 Continuous Inputs (15.5 ~ 25.0)
        // ---------------------------------------------------------
        $display("[Time %t] Phase 3: Sending 20 Data items...", $time);

        for (k = 31; k <= 50; k = k + 1) begin
            send_data(k * 0.5); // 15.5, 16.0 ... 25.0
        end

        // ---------------------------------------------------------
        // Phase 4: Finish
        // ---------------------------------------------------------
        @(posedge i_clk);
        i_valid <= 0;
        i_variance <= 0;

        // Wait for pipeline to flush (Latency ~10 cycles + Margin)
        #500;
        
        $display("----------------------------------------------------------------");
        $display(" Simulation Finished");
        $display("----------------------------------------------------------------");
        $finish;
    end

    // Helper Task: Send Data per 1 Clock
    task send_data;
        input real val;
        reg [15:0] fixed_val;
        begin
            fixed_val = val * 2048; // Convert Real to UQ5.11
            
            @(posedge i_clk);
            i_valid    <= 1;
            i_variance <= fixed_val;
        end
    endtask

    // =================================================================
    // 5. Output Monitor
    // =================================================================
    always @(posedge i_clk) begin
        if (o_valid) begin
            // Print only selected ranges to keep log clean, or print all
            // For verification, it's good to see the input vs output trend
            $display("[Time %t] Out Valid | Result: %f (Hex: %h)", 
                     $time, real_output, o_result);
        end
    end

endmodule