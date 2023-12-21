from ftgemm import FTGEMM
import torch

gemm = FTGEMM()
a = torch.ones((32, 64), dtype=torch.int8)
b = torch.ones((32, 64), dtype=torch.int8)
c = 1.0
int8_out = gemm.linear_a8_w8_o8(a.cuda(), b.cuda(), c)
int32_out = gemm.linear_a8_w8_o32(a.cuda(), b.cuda())
print("int8_out:", int8_out)
print("int32_out:", int32_out)
