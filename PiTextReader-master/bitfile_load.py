from pynq import Overlay

bitfile = "/home/xilinx/finn-accel.bit"
overlay = Overlay(bitfile)
overlay.download()

print("FPGA Bitfile successfully loaded!")
