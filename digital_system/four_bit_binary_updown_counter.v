module up_down_counter(
    input clk,
    input resetn,
    input sel,
    output reg [3:0] q
);
    wire clk_1hz;
    clock_divider #(49999999) div1(clk, clk_1hz);

    // simulation ^= clk, synthesis ^= clk_1hz
    always @(negedge clk_1hz, negedge resetn) begin
        if (~resetn)
            q <= 0;
        else begin
            case (sel)
                1'b0: q <= q + 1'b1;
                1'b1: q <= q - 1'b1;
            endcase
        end
    end
endmodule

-------------------------------------------------------

module clock_divider #(
    parameter div = 49999999
)(
    input clk_in,
    output reg clk_out
);

    reg [25:0] q;

    initial begin
        q <= 0;
        clk_out = 0;
    end

    always @(posedge clk_in) begin
        if (q == div) begin
            clk_out <= ~clk_out;
            q <= 0;
        end
        else 
            q <= q + 1;
    end
endmodule


-----------------------------------------------------

module up_down_counter_tb();
    reg clk, resetn, sel;
    wire [3:0] q;

    up_down_counter uut(clk, resetn, sel, q);

    initial begin
        clk = 0; 
        resetn = 0; 
        sel = 0;
        #20 resetn = 1;
    end

    always #25 clk = ~clk;
    always #500 sel = ~sel;
endmodule

