# coding=utf-8
# f t cent?
import torch
import time
import random

if __name__ == '__main__':
    while True:
        dim = 3000
        b = torch.Tensor(1300, dim).cuda()
        c = torch.Tensor(dim, 1300).cuda()
        for i in range(200):
            d = torch.matmul(b, c)
        time.sleep(0.0001)
