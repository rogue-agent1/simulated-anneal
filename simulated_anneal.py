#!/usr/bin/env python3
"""Simulated annealing optimizer. Zero dependencies."""
import math, random

def simulated_annealing(objective, neighbor, initial, temp=1000, cooling=0.995, min_temp=1e-8, seed=42):
    random.seed(seed)
    current = initial; current_cost = objective(current)
    best = current; best_cost = current_cost
    t = temp; history = [(current, current_cost)]
    while t > min_temp:
        candidate = neighbor(current)
        candidate_cost = objective(candidate)
        delta = candidate_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta/t):
            current = candidate; current_cost = candidate_cost
        if current_cost < best_cost:
            best = current; best_cost = current_cost
        t *= cooling; history.append((best, best_cost))
    return best, best_cost, history

def tsp_anneal(cities, temp=10000, cooling=0.9995, seed=42):
    def distance(route):
        d = 0
        for i in range(len(route)):
            j = (i+1) % len(route)
            d += math.hypot(cities[route[i]][0]-cities[route[j]][0], cities[route[i]][1]-cities[route[j]][1])
        return d
    def swap_neighbor(route):
        r = route[:]
        i, j = sorted(random.sample(range(len(r)), 2))
        r[i:j+1] = reversed(r[i:j+1])
        return r
    initial = list(range(len(cities)))
    best, cost, _ = simulated_annealing(distance, swap_neighbor, initial, temp, cooling, seed=seed)
    return best, cost

if __name__ == "__main__":
    best, cost, _ = simulated_annealing(lambda x: (x-3)**2, lambda x: x+random.gauss(0,1), 10.0)
    print(f"Min: x={best:.4f}, cost={cost:.6f}")
