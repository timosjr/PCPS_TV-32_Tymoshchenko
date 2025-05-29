import unittest
from main import (
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    order_crossover,
    mutate,
    evaluate_population,
    next_generation,
    get_best_individual
)

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.cities = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.population = initialize_population(10, self.cities)
        self.fitnesses = evaluate_population(self.population)

    def test_initialize_population(self):
        self.assertEqual(len(self.population), 10)
        for individual in self.population:
            self.assertCountEqual(individual, self.cities)

    def test_evaluate_fitness(self):
        ind = [(0, 0), (0, 1), (1, 1), (1, 0)]
        fitness = evaluate_fitness(ind)
        self.assertGreater(fitness, 0)

    def test_tournament_selection(self):
        parent = tournament_selection(self.population, self.fitnesses)
        self.assertIn(parent, self.population)

    def test_order_crossover(self):
        parent1 = self.population[0]
        parent2 = self.population[1]
        child = order_crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        self.assertCountEqual(child, parent1)

    def test_mutate(self):
        individual = self.population[0][:]
        mutate(individual, mutation_rate=1.0)  # Force mutation
        self.assertCountEqual(individual, self.cities)

    def test_evaluate_population(self):
        fitnesses = evaluate_population(self.population)
        self.assertEqual(len(fitnesses), len(self.population))
        self.assertTrue(all(isinstance(f, float) for f in fitnesses))

    def test_next_generation(self):
        next_pop = next_generation(self.population, self.fitnesses, elite_size=2)
        self.assertEqual(len(next_pop), len(self.population))
        for ind in next_pop:
            self.assertCountEqual(ind, self.cities)

    def test_get_best_individual(self):
        best, dist = get_best_individual(self.population)
        self.assertIn(best, self.population)
        self.assertIsInstance(dist, float)

if __name__ == '__main__':
    unittest.main()
