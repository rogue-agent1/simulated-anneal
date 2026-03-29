#!/usr/bin/env python3
"""simulated_anneal - Simulated annealing optimization framework."""
import sys, math, random

def simulated_annealing(initial, energy_fn, neighbor_fn, temp=1000, cooling=0.995, min_temp=0.01, seed=42):
    rng = random.Random(seed)
    current = initial
    current_energy = energy_fn(current)
    best = current
    best_energy = current_energy
    t = temp
    history = []
    while t > min_temp:
        candidate = neighbor_fn(current, rng)
        candidate_energy = energy_fn(candidate)
        delta = candidate_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / t):
            current = candidate
            current_energy = candidate_energy
        if current_energy < best_energy:
            best = current
            best_energy = current_energy
        history.append(best_energy)
        t *= cooling
    return best, best_energy, history

def test():
    def sphere(x):
        return sum(xi**2 for xi in x)
    def sphere_neighbor(x, rng):
        i = rng.randint(0, len(x)-1)
        new = list(x)
        new[i] += rng.gauss(0, 1)
        return new
    best, energy, history = simulated_annealing(
        [5.0, 5.0, 5.0], sphere, sphere_neighbor, temp=100, cooling=0.99)
    assert energy < 1.0
    assert all(abs(x) < 2 for x in best)
    assert history[-1] <= history[0]
    def rastrigin(x):
        return 10*len(x) + sum(xi**2 - 10*math.cos(2*math.pi*xi) for xi in x)
    best2, e2, _ = simulated_annealing(
        [3.0, -3.0], rastrigin, sphere_neighbor, temp=500, cooling=0.998)
    assert e2 < 20
    best3, e3, _ = simulated_annealing([0.0], sphere, sphere_neighbor, temp=1)
    assert e3 < 5
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("simulated_anneal: Simulated annealing. Use --test")
