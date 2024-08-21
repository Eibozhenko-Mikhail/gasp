import numpy as np
k= 100
l = 0
counter = 0
for n in range(k):
    p = int(np.power(2,4*n/k)//1)
    if n==l:
        print("update epoch", n)
        l += p
        counter+=1

print(counter)