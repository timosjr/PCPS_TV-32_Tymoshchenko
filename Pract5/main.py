import random
import math

# Ініціалізація популяції
def initialize_population(num_individuals, cities):
    population = []
    for _ in range(num_individuals):
        individual = cities[:]
        random.shuffle(individual)
        population.append(individual)
    return population

# Обчислення придатності (загальної довжини маршруту)
def evaluate_fitness(individual):
    distance = 0.0
    for i in range(len(individual)):
        city_a = individual[i]
        city_b = individual[(i + 1) % len(individual)]
        distance += math.dist(city_a, city_b)
    return distance

# Турнірний відбір
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0][:]

# Кросовер (Order Crossover)
def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    hole = parent2[a:b]
    child = [None] * size
    child[a:b] = hole
    pos = b
    for city in parent1:
        if city not in hole:
            if pos >= size:
                pos = 0
            child[pos] = city
            pos += 1
    return child

# Мутація (обмін містами)
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]

# Обчислення придатності популяції
def evaluate_population(population):
    return [evaluate_fitness(ind) for ind in population]

# Створення нового покоління (з елітністю)
def next_generation(population, fitnesses, elite_size):
    new_population = []
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])]
    elites = sorted_pop[:elite_size]
    new_population.extend(elites)
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = order_crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    return new_population

# Пошук найкращого індивіда
def get_best_individual(population):
    fitnesses = evaluate_population(population)
    min_index = fitnesses.index(min(fitnesses))
    return population[min_index], fitnesses[min_index]

# Основний цикл генетичного алгоритму
def run_ga(cities, num_individuals=50, max_iterations=100, elite_size=2, stop_if_no_improvement=True, patience=20):
    population = initialize_population(num_individuals, cities)
    best, best_distance = get_best_individual(population)
    no_improvement = 0

    for iteration in range(max_iterations):
        fitnesses = evaluate_population(population)
        current_best_distance = min(fitnesses)
        print(f"Ітерація {iteration + 1}: найкраща відстань = {current_best_distance:.2f}")

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best = population[fitnesses.index(current_best_distance)]
            no_improvement = 0
        else:
            no_improvement += 1

        if stop_if_no_improvement and no_improvement >= patience:
            print(f"Раннє завершення після {iteration + 1} ітерацій через відсутність покращення.")
            break

        population = next_generation(population, fitnesses, elite_size)

    print("\nНайкращий знайдений маршрут:")
    for i, city in enumerate(best):
        print(f"{i + 1}: {city}")
    print(f"Загальна довжина маршруту: {best_distance:.2f}")
    return best, best_distance

if __name__ == "__main__":
    try:
        num_cities = int(input("Введіть кількість міст: "))
        num_individuals = int(input("Введіть розмір популяції: "))
        max_iterations = int(input("Введіть максимальну кількість ітерацій: "))
        elite_size = int(input("Введіть розмір еліти (наприклад, 2): "))
        dynamic_stop = input("Зупинятись, якщо немає покращення? (так/ні): ").strip().lower() == 'так'

        if dynamic_stop:
            patience = int(input("Скільки ітерацій чекати без покращення перед зупинкою: "))
        else:
            patience = 0

        cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_cities)]

        run_ga(
            cities,
            num_individuals=num_individuals,
            max_iterations=max_iterations,
            elite_size=elite_size,
            stop_if_no_improvement=dynamic_stop,
            patience=patience
        )

    except ValueError:
        print("Помилка: введіть коректні числові значення.")
