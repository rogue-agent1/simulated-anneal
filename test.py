from simulated_anneal import simulated_annealing, tsp_anneal
import random
best, cost, hist = simulated_annealing(
    lambda x: (x-5)**2,
    lambda x: x+random.gauss(0,0.5),
    0.0, temp=100, cooling=0.99, seed=42)
assert abs(best-5) < 1.0
assert len(hist) > 10
cities = [(0,0),(1,0),(1,1),(0,1)]
route, dist = tsp_anneal(cities, temp=1000, cooling=0.995, seed=42)
assert len(route) == 4
assert dist <= 4.5  # optimal is 4.0 for a square
print("simulated_anneal tests passed")
