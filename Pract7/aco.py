import numpy as np
import random
from graph import Graph


class ACO:
    def __init__(self, graph: Graph, num_ants, num_iterations, num_vehicles,
                 alpha=1.0, beta=2.0, evaporation_rate=0.5, initial_pheromone=1.0, Q=100):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_vehicles = num_vehicles
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        self.pheromones = np.full((graph.num_nodes, graph.num_nodes), initial_pheromone, dtype=float)
        np.fill_diagonal(self.pheromones, 0)

        self.best_solution_routes = None
        self.best_path_length = np.inf

        self.depot_node = 0  # Депо - це вузол 0
        self.all_client_nodes = set(range(self.graph.num_nodes)) - {self.depot_node}

    def _calculate_total_solution_length(self, solution_routes):
        total_length = 0
        visited_clients_check = set()

        for route in solution_routes:
            if not route or len(route) < 2 or route[0] != self.depot_node or route[-1] != self.depot_node:
                return np.inf

            route_length = 0
            for i in range(len(route) - 1):
                node1, node2 = route[i], route[i + 1]
                dist = self.graph.get_distance(node1, node2)
                route_length += dist
                if node2 != self.depot_node:
                    visited_clients_check.add(node2)
            total_length += route_length


        if visited_clients_check != self.all_client_nodes:
            return np.inf

        return total_length

    def _select_next_node(self, current_node, unvisited_clients_for_current_vehicle, total_unvisited_clients):
        potential_next_nodes_and_reasons = []

        for client_node in total_unvisited_clients:
            if client_node != self.depot_node:
                potential_next_nodes_and_reasons.append((client_node, False))

        if current_node != self.depot_node and (
                len(unvisited_clients_for_current_vehicle) > 0 or not total_unvisited_clients):
            potential_next_nodes_and_reasons.append((self.depot_node, True))
        elif current_node == self.depot_node and not total_unvisited_clients:
            potential_next_nodes_and_reasons.append((self.depot_node, True))

        if not potential_next_nodes_and_reasons:
            return None

        probabilities = []
        total_probability = 0

        for neighbor, is_depot_return in potential_next_nodes_and_reasons:
            pheromone = self.pheromones[current_node, neighbor]
            distance = self.graph.get_distance(current_node, neighbor)

            heuristic = 1.0 / (distance + 1e-9) if distance > 0 else 0.0

            depot_return_bonus = 1.0
            if is_depot_return:
                if not total_unvisited_clients:
                    depot_return_bonus = 5.0
                elif len(
                        unvisited_clients_for_current_vehicle) > 0:
                    depot_return_bonus = 1.0

            probability = (pheromone ** self.alpha) * (heuristic ** self.beta) * depot_return_bonus
            probabilities.append(probability)
            total_probability += probability

        if total_probability == 0:
            # Якщо всі ймовірності нульові, обираємо випадково з доступних
            return random.choice([node for node, _ in potential_next_nodes_and_reasons])

        normalized_probabilities = [p / total_probability for p in probabilities]
        chosen_tuple = random.choices(potential_next_nodes_and_reasons, weights=normalized_probabilities, k=1)[0]
        return chosen_tuple[0]

    def _construct_solution_routes(self):
        solution_routes = []
        clients_to_visit_overall = set(self.all_client_nodes)

        for vehicle_idx in range(self.num_vehicles):
            current_route = [self.depot_node]
            current_node = self.depot_node

            clients_visited_by_this_vehicle = set()

            while True:
                if not clients_to_visit_overall and current_node == self.depot_node:
                    # Якщо ми в депо і всі клієнти вже відвідані (іншими ТЗ)
                    # цей ТЗ може просто повернутися "порожнім"
                    break

                next_node = self._select_next_node(current_node, clients_visited_by_this_vehicle,
                                                   clients_to_visit_overall)

                if next_node is None:
                    # Якщо немає куди рухатися (наприклад, застрягли)
                    break

                if next_node == self.depot_node:
                    if current_node != self.depot_node:
                        current_route.append(self.depot_node)
                    break

                current_route.append(next_node)
                current_node = next_node

                clients_to_visit_overall.discard(current_node)
                clients_visited_by_this_vehicle.add(current_node)

                if not clients_to_visit_overall:
                    if current_node != self.depot_node:
                        current_route.append(self.depot_node)
                    break

            if current_route[0] == self.depot_node and current_route[-1] == self.depot_node:
                solution_routes.append(current_route)
            else:
                # Якщо маршрут не замкнений, це помилка, і все рішення мурахи недійсне
                # print(f"Мураха {vehicle_idx}: Маршрут не замкнений: {current_route}") # Debug
                return None

        if clients_to_visit_overall:
            return None

        return solution_routes

    def _update_pheromones(self, ant_solutions):
        self.pheromones *= (1 - self.evaporation_rate)

        for solution_routes, total_length in ant_solutions:
            if total_length != np.inf and solution_routes is not None:
                pheromone_deposit = self.Q / total_length
                for route in solution_routes:
                    for i in range(len(route) - 1):
                        node1, node2 = route[i], route[i + 1]
                        self.pheromones[node1, node2] += pheromone_deposit
                        self.pheromones[node2, node1] += pheromone_deposit

    def run(self, start_node=0, update_callback=None):
        if start_node != self.depot_node:
            print(
                f"Попередження: Для VRP стартова точка (депо) має бути {self.depot_node}. Встановлюємо start_node = {self.depot_node}.")
            start_node = self.depot_node

        self.best_solution_routes = None
        self.best_path_length = np.inf

        for iteration in range(self.num_iterations):
            ant_solutions = []
            failed_ant_count = 0

            for ant_idx in range(self.num_ants):
                solution_routes = self._construct_solution_routes()

                if solution_routes is None:
                    failed_ant_count += 1
                    continue

                total_length = self._calculate_total_solution_length(solution_routes)

                if total_length == np.inf:
                    failed_ant_count += 1
                    continue

                ant_solutions.append((solution_routes, total_length))

                if total_length < self.best_path_length:
                    self.best_path_length = total_length
                    self.best_solution_routes = solution_routes

            # Обновлення феромонів
            if ant_solutions:
                self._update_pheromones(ant_solutions)
            else:
                self.pheromones *= (1 - self.evaporation_rate)

            if update_callback:
                update_callback(iteration, self.best_solution_routes, self.best_path_length, self.pheromones)

            print(
                f"Ітерація {iteration + 1}: Невдалих мурах: {failed_ant_count}/{self.num_ants}. Найкраща загальна довжина = {self.best_path_length:.2f}")

        return self.best_solution_routes, self.best_path_length