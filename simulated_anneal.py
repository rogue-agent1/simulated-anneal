import argparse, random, math

def sphere(x): return sum(v**2 for v in x)
def rosenbrock(x): return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
def ackley(x):
    n = len(x)
    return -20*math.exp(-0.2*math.sqrt(sum(v**2 for v in x)/n)) - math.exp(sum(math.cos(2*math.pi*v) for v in x)/n) + 20 + math.e

FUNCS = {"sphere": sphere, "rosenbrock": rosenbrock, "ackley": ackley}

def sa(func, dim=5, t0=100, tf=0.01, cool=0.995, seed=None):
    if seed: random.seed(seed)
    x = [random.uniform(-5, 5) for _ in range(dim)]
    fx = func(x)
    best, fbest = x[:], fx
    t = t0
    step = 0
    while t > tf:
        nx = [xi + random.gauss(0, t*0.1) for xi in x]
        nfx = func(nx)
        if nfx < fx or random.random() < math.exp(-(nfx - fx) / t):
            x, fx = nx, nfx
        if fx < fbest: best, fbest = x[:], fx
        t *= cool
        step += 1
        if step % 500 == 0: print(f"Step {step:5d} T={t:.4f} best={fbest:.6f}")
    print(f"Final: best={fbest:.6f} steps={step}")
    return best, fbest

def main():
    p = argparse.ArgumentParser(description="Simulated annealing")
    p.add_argument("func", choices=FUNCS.keys())
    p.add_argument("-d", "--dim", type=int, default=5)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    sa(FUNCS[args.func], args.dim, seed=args.seed)

if __name__ == "__main__":
    main()
