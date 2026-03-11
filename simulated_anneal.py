#!/usr/bin/env python3
"""Simulated annealing optimizer."""
import sys, random, math
random.seed(42)
def rastrigin(x): return sum(xi**2-10*math.cos(2*math.pi*xi)+10 for xi in x)
dim=int(sys.argv[1]) if len(sys.argv)>1 else 5
x=[random.uniform(-5,5) for _ in range(dim)]; best_x=x[:]; best_f=f=rastrigin(x)
T=100; cool=0.995
for i in range(10000):
    nx=[xi+random.gauss(0,T*0.01) for xi in x]; nf=rastrigin(nx)
    if nf<f or random.random()<math.exp(-(nf-f)/max(T,1e-10)):
        x,f=nx,nf
        if f<best_f: best_x,best_f=x[:],f
    T*=cool
    if i%1000==0: print(f"Step {i}: T={T:.2f}, f={f:.4f}, best={best_f:.4f}")
print(f"\nOptimum: f={best_f:.6f}")
print(f"Solution: [{', '.join(f'{x:.4f}' for x in best_x)}]")
