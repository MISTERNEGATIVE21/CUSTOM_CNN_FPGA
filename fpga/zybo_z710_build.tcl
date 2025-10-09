# Vivado batch TCL to create a Zybo Z7-10 design with PS7 + AXI DMA + HLS IP and generate bitstream
# How to run (batch):
#   vivado -mode batch -nolog -nojournal -notrace \
#     -source fpga/zybo_z710_build.tcl -tclargs /abs/path/to/hls_ip_repo myproject /abs/path/to/build_out
#
# Arguments (via -tclargs) or pre-set variables:
#   ip_repo  -> e.g., hls4ml_prj/myproject_prj/solution1/impl/ip
#   top_name -> HLS IP top name (default: myproject)
#   out_dir  -> Output directory for .bit/.hwh (default: ./vivado_out)

# Accept -tclargs if provided
if {[info exists argv] && [llength $argv] > 0} {
    if {[llength $argv] >= 1} { set ip_repo  [lindex $argv 0] }
    if {[llength $argv] >= 2} { set top_name [lindex $argv 1] }
    if {[llength $argv] >= 3} { set out_dir  [lindex $argv 2] }
}

if {![info exists ip_repo]} { puts "ERROR: ip_repo not set"; exit 1 }
if {![info exists top_name]} { set top_name "myproject" }
if {![info exists out_dir]} { set out_dir "[pwd]/vivado_out" }

file mkdir $out_dir

create_project plant_detect $out_dir/plant_detect -part xc7z010clg400-1 -force
set_property target_language VHDL [current_project]

# Add HLS IP repo
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# Create block design
create_bd_design "system"

# Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config { make_external "FIXED_IO, DDR" } [get_bd_cells processing_system7_0]

# AXI DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list CONFIG.c_include_mm2s_dre {1} CONFIG.c_include_s2mm_dre {1} CONFIG.c_include_sg {0}] [get_bd_cells axi_dma_0]

# AXI Interconnect for control
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

# HLS IP from repo (assumes vlnv xilinx.com:hls:<top_name>:1.0)
set vlnv "xilinx.com:hls:${top_name}:1.0"
create_bd_cell -type ip -vlnv $vlnv ${top_name}_0

# Clocking and resets
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_interconnect_0/S00_AXI} Clk {/processing_system7_0/FCLK_CLK0} }  [get_bd_intf_pins axi_interconnect_0/S00_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master {/axi_interconnect_0/M00_AXI} Slave {/${top_name}_0/s_axi_CONTROL_BUS} Clk {/processing_system7_0/FCLK_CLK0} }  [get_bd_intf_pins ${top_name}_0/s_axi_CONTROL_BUS]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master {/axi_interconnect_0/M01_AXI} Slave {/axi_dma_0/S_AXI_LITE} Clk {/processing_system7_0/FCLK_CLK0} }  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
# Memory-mapped connections for DMA
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/M_AXI_MM2S} Clk {/processing_system7_0/FCLK_CLK0} }  [get_bd_intf_pins axi_dma_0/M_AXI_MM2S]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/M_AXI_S2MM} Clk {/processing_system7_0/FCLK_CLK0} }  [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

# Stream connections: assumes HLS IP has input axis port named 'in_r' and output 'out_r'. Adjust if different.
# Try common hls4ml names: input_1, output_1; otherwise in_r/out_r
set in_port  "${top_name}_0/INPUT_STREAM"
set out_port "${top_name}_0/OUTPUT_STREAM"
if {[llength [get_bd_intf_pins -quiet ${top_name}_0/input_1]]} { set in_port  "${top_name}_0/input_1" }
if {[llength [get_bd_intf_pins -quiet ${top_name}_0/output_1]]} { set out_port "${top_name}_0/output_1" }
if {[llength [get_bd_intf_pins -quiet ${top_name}_0/in_r]]} { set in_port  "${top_name}_0/in_r" }
if {[llength [get_bd_intf_pins -quiet ${top_name}_0/out_r]]} { set out_port "${top_name}_0/out_r" }

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins $in_port]
connect_bd_intf_net [get_bd_intf_pins $out_port] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Clock and resets to stream and IP
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0} Rst {/processing_system7_0/FCLK_RESET0_N} } [get_bd_pins axi_dma_0/mm2s_cntrl_reset_out_n]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0} Rst {/processing_system7_0/FCLK_RESET0_N} } [get_bd_pins axi_dma_0/s2mm_cntrl_reset_out_n]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0} Rst {/processing_system7_0/FCLK_RESET0_N} } [get_bd_pins ${top_name}_0/ap_clk]

# Make DMA stream resets
if {[llength [get_bd_pins -quiet axi_dma_0/axi_resetn]]} { connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_dma_0/axi_resetn] }

validate_bd_design
save_bd_design

make_wrapper -files [get_files $out_dir/plant_detect/plant_detect.srcs/sources_1/bd/system/system.bd] -top
add_files -norecurse $out_dir/plant_detect/plant_detect.srcs/sources_1/bd/system/hdl/system_wrapper.v
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Export bitstream and hwh
set bitfile $out_dir/plant_detect/plant_detect.runs/impl_1/system_wrapper.bit
set hwhfile $out_dir/plant_detect/plant_detect.gen/sources_1/bd/system/hw_handoff/system.hwh
if {[file exists $bitfile]} { file copy -force $bitfile $out_dir/system.bit }
if {[file exists $hwhfile]} { file copy -force $hwhfile $out_dir/system.hwh }

puts "DONE: Bitstream at $out_dir/system.bit, HWH at $out_dir/system.hwh"
