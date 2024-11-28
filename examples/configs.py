import os
spad_addr = 0x1000000000000000
packet_size = 32
stride = 256
ndp_units = 32
data_size = 4
intensity = int(os.getenv("NDP_INTENSITY", default="1"))
unroll_level = int(os.getenv("NDP_UNROLL_LEVEL", default="1"))