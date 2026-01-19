import numpy as np
import math
# A1
def manual_dot(n):
    A=np.random.rand(n)
    B=np.random.rand(n)
    dot_pro=0
    for i in range(n):
        dot_pro=dot_pro+A[i]*B[i]
    return dot_pro
    
def manual_norm_A(n):
    A=np.random.rand(n)
    total_A=0
    for i in range(n):
        total_A=total_A+(A[i]*A[i])
    return total_A
        
def manual_norm_B(n):
    B=np.random.rand(n)
    total_B=0
    for i in range(n):
        total_B=total_B+(B[i]*B[i])
    return total_B
    
def compare_manual_package(n):
    A=np.random.rand(n)
    B=np.random.rand(n)
    if manual_dot(n) <= np.dot(A,B):
        return "Package dot is efficient"
    if manual_norm_A(n) <= np.linalg.norm(A):
        return "Package norm_A is efficient"
    if manual_norm_B(n) <= np.linalg.norm(B):
        return "Package norm_B is efficient"

if __name__ == "__main__":
    n=5
    print(compare_manual_package(n))