from pynq import Overlay

bitfile = "/home/ubuntu/finn-cybsec-mlp-demo/bitfile/finn-accel.bit"
overlay = Overlay(bitfile)
overlay.download()

print("FPGA Bitfile successfully loaded!")
