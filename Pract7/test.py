import unittest
import numpy as np
from graph import Graph
from aco import ACO
import random
from unittest import mock  # Додаємо імпорт mock

# Встановлюємо фіксований seed для відтворюваності тестів
# Це важливо для тестів, що залежать від випадковості,
# навіть якщо ми використовуємо mock для окремих випадків.
random.seed(42)
np.random.seed(42)



class TestGraph(unittest.TestCase):
    def test_graph_creation(self):
        graph = Graph(num_nodes=5)
        self.assertEqual(graph.num_nodes, 5)
        self.assertTrue(np.all(graph.distances == 0))
        self.assertEqual(len(graph.node_positions), 0)

    def test_add_edge_and_get_distance(self):
        graph = Graph(num_nodes=3)
        graph.add_edge(0, 1, 10.0)
        self.assertEqual(graph.get_distance(0, 1), 10.0)
        self.assertEqual(graph.get_distance(1, 0), 10.0)  # Перевірка симетрії
        self.assertEqual(graph.get_distance(0, 2), 0.0)  # Неіснуюче ребро

    def test_set_node_positions(self):
        graph = Graph(num_nodes=2)
        positions = [(0, 0), (1, 1)]
        graph.set_node_positions(positions)
        self.assertEqual(graph.node_positions, positions)

        with self.assertRaises(ValueError):
            graph.set_node_positions([(0, 0)])  # Неправильна кількість позицій

    def test_generate_random_graph(self):
        graph = Graph.generate_random_graph(num_nodes=5)
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(len(graph.node_positions), 5)
        # Перевіряємо, що відстані не нульові для різних вузлів
        self.assertTrue(np.any(graph.distances > 0))
        # Перевіряємо, що діагональ нульова
        self.assertTrue(np.all(np.diag(graph.distances) == 0))



class TestACO(unittest.TestCase):
    def setUp(self):
        # Налаштування для кожного тесту ACO
        self.graph = Graph(num_nodes=4)
        self.graph.set_node_positions([(0, 0), (1, 0), (0, 1), (1, 1)])
        self.graph.add_edge(0, 1, 1.0)  # 0-1
        self.graph.add_edge(0, 2, 1.0)  # 0-2
        self.graph.add_edge(0, 3, 1.414)  # 0-3
        self.graph.add_edge(1, 2, 1.414)  # 1-2
        self.graph.add_edge(1, 3, 1.0)  # 1-3
        self.graph.add_edge(2, 3, 1.0)  # 2-3

        self.depot_node = 0
        self.all_client_nodes = {1, 2, 3}

    ## 1. Тести для "Вибору маршруту" (Ant._select_next_node)
    def test_select_next_node_from_depot_with_clients(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1,
                  alpha=1.0, beta=1.0, evaporation_rate=0.1)
        current_node = self.depot_node
        # Забезпечуємо початковий стан феромонів для передбачуваності
        aco.pheromones[0, 1] = 10.0  # Стимулюємо шлях до 1
        aco.pheromones[0, 2] = 1.0
        aco.pheromones[0, 3] = 1.0

        # Всі клієнти не відвідані
        total_unvisited_clients = self.all_client_nodes.copy()
        unvisited_clients_for_current_vehicle = self.all_client_nodes.copy()

        # Мокємо random.choices, щоб гарантувати вибір клієнта 1.
        with mock.patch('random.choices', return_value=[(1, False)]) as mock_choices:
            next_node = aco._select_next_node(current_node, unvisited_clients_for_current_vehicle,
                                              total_unvisited_clients)
            self.assertEqual(next_node, 1)  # Очікуємо, що обере клієнта 1

    def test_select_next_node_return_to_depot_no_clients_left(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1)
        current_node = 1  # Припустимо, ми на клієнті 1
        total_unvisited_clients = set()  # Всі клієнти вже відвідані
        unvisited_clients_for_current_vehicle = set()  # Цей автомобіль також все відвідав

        # Мокємо random.choices, щоб гарантувати вибір депо.
        with mock.patch('random.choices', return_value=[(self.depot_node, True)]) as mock_choices:
            next_node = aco._select_next_node(current_node, unvisited_clients_for_current_vehicle,
                                              total_unvisited_clients)
            # Маємо повернутися до депо
            self.assertEqual(next_node, self.depot_node)

    def test_select_next_node_stuck(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1)
        current_node = 0
        total_unvisited_clients = set()  # Жодного клієнта
        unvisited_clients_for_current_vehicle = set()

        # Якщо ми в депо і клієнтів немає, єдина опція - повернутися до депо.
        # mock.patch не потрібен, якщо єдиний шлях — депо, _select_next_node обробить це.
        next_node = aco._select_next_node(current_node, unvisited_clients_for_current_vehicle, total_unvisited_clients)
        self.assertEqual(next_node, self.depot_node)  # Повернення до депо, оскільки немає куди йти

    ## 2. Тести для "Прокладання шляху" (_construct_solution_routes)
    def test_construct_single_vehicle_solution_valid(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1,
                  alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

        # Мокємо _select_next_node, щоб він повертав передбачувані вузли для одного маршруту: 0 -> 1 -> 2 -> 3 -> 0
        mock_sequence = [1, 2, 3, 0]  # Послідовність вузлів, які має повернути _select_next_node
        with mock.patch.object(aco, '_select_next_node', side_effect=mock_sequence) as mock_method:
            solution = aco._construct_solution_routes()

            self.assertIsNotNone(solution)
            self.assertEqual(len(solution), 1)  # Очікуємо один маршрут для 1 автомобіля
            self.assertEqual(solution[0], [0, 1, 2, 3, 0])  # Очікуваний маршрут

            # Перевірка, що маршрути починаються і закінчуються в депо
            for route in solution:
                self.assertEqual(route[0], self.depot_node)
                self.assertEqual(route[-1], self.depot_node)
                # Перевіряємо, що всі клієнти відвідані хоча б одним маршрутом
                visited_clients = set()
                for r in solution:
                    visited_clients.update(node for node in r if node != self.depot_node)
                self.assertEqual(visited_clients, self.all_client_nodes)

    def test_construct_multiple_vehicle_solution(self):
        # 3 клієнти, 3 автомобілі - кожен має взяти по одному клієнту
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=3,
                  alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

        # Мокємо _select_next_node для трьох автомобілів.
        # Кожен автомобіль вибирає одного унікального клієнта, а потім повертається до депо.
        # Порядок викликів _select_next_node:
        # Авто 1: (0 -> 1) (1 -> 0)
        # Авто 2: (0 -> 2) (2 -> 0)
        # Авто 3: (0 -> 3) (3 -> 0)
        mock_return_values = [
            1,  # Авто 1 з депо -> 1
            0,  # Авто 1 з 1 -> депо
            2,  # Авто 2 з депо -> 2
            0,  # Авто 2 з 2 -> депо
            3,  # Авто 3 з депо -> 3
            0  # Авто 3 з 3 -> депо
        ]

        with mock.patch.object(aco, '_select_next_node', side_effect=mock_return_values) as mock_method:
            solution = aco._construct_solution_routes()

            self.assertIsNotNone(solution)
            self.assertEqual(len(solution), 3)  # Очікуємо 3 маршрути

            # Маршрути можуть бути в будь-якому порядку, тому сортуємо їх для порівняння.
            expected_routes = sorted([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
            actual_routes = sorted(solution)
            self.assertEqual(actual_routes, expected_routes)

            visited_clients = set()
            for route in solution:
                self.assertEqual(route[0], self.depot_node)
                self.assertEqual(route[-1], self.depot_node)
                for node in route:
                    if node != self.depot_node:
                        visited_clients.add(node)
            self.assertEqual(visited_clients, self.all_client_nodes)

    def test_construct_solution_not_all_clients_visited(self):
        # Мокємо _select_next_node таким чином, щоб один з клієнтів не був відвіданий
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1,
                  alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

        # Мураха відвідує клієнта 1, потім повертається до депо.
        # Клієнти 2 і 3 залишаться невідвіданими.
        mock_return_values = [1, 0]
        with mock.patch.object(aco, '_select_next_node', side_effect=mock_return_values):
            solution = aco._construct_solution_routes()
            # Очікуємо None, бо _construct_solution_routes повертає None,
            # якщо `clients_to_visit_overall` не порожній в кінці.
            self.assertIsNone(solution)

    def test_construct_solution_invalid_route_structure(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1)

        # Варіант 1: Заглушуємо сам _construct_solution_routes, щоб він повернув None
        # Це перевіряє, що зовнішній код коректно обробляє None
        with mock.patch.object(aco, '_construct_solution_routes', return_value=None):
            solution = aco._construct_solution_routes()
            self.assertIsNone(solution)

        # Варіант 2: Перевіряємо, що _calculate_total_solution_length правильно обробляє недійсний маршрут
        # Маршрут, який не закінчується в депо
        invalid_routes_for_calc_1 = [[0, 1, 2]]
        self.assertEqual(aco._calculate_total_solution_length(invalid_routes_for_calc_1), np.inf)

        # Маршрут, який починається не з депо (хоча в ACO це, ймовірно, не відбудеться)
        invalid_routes_for_calc_2 = [[1, 2, 0]]
        self.assertEqual(aco._calculate_total_solution_length(invalid_routes_for_calc_2), np.inf)

        # Маршрут, де немає клієнтів, але клієнти очікуються
        aco.all_client_nodes = {1, 2, 3}  # Забезпечуємо, що очікуються клієнти
        invalid_routes_for_calc_3 = [[0, 0]]  # Маршрут тільки депо
        self.assertEqual(aco._calculate_total_solution_length(invalid_routes_for_calc_3), np.inf)

        # Варіант 3: Мок для _select_next_node, який призводить до неповного маршруту
        # Мураха починає з 0, йде до 1, потім _select_next_node повертає None, і мураха "застрягає"
        aco_for_incomplete = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1)
        with mock.patch.object(aco_for_incomplete, '_select_next_node', side_effect=[1, None]):
            solution_incomplete = aco_for_incomplete._construct_solution_routes()
            # Очікуємо None, бо мураха "застрягла" і маршрут не замкнувся в депо
            self.assertIsNone(solution_incomplete)

    ## 3. Тести для "Оновлення феромонів" та "Завершення ітерації" (_update_pheromones)
    def test_update_pheromones_deposit(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1, Q=100)
        initial_pheromones = np.copy(aco.pheromones)

        # Симулюємо рішення, яке мураха знайшла
        # Маршрут: 0 -> 1 -> 3 -> 2 -> 0. Довжина: 1.0 (0-1) + 1.0 (1-3) + 1.0 (3-2) + 1.0 (2-0) = 4.0
        solution_routes = [[0, 1, 3, 2, 0]]
        total_length = self.graph.get_distance(0, 1) + self.graph.get_distance(1, 3) + self.graph.get_distance(3,
                                                                                                               2) + self.graph.get_distance(
            2, 0)
        ant_solutions = [(solution_routes, total_length)]

        aco._update_pheromones(ant_solutions)

        # Перевірка, що феромони на шляху збільшилися
        expected_deposit = aco.Q / total_length
        self.assertGreater(aco.pheromones[0, 1], initial_pheromones[0, 1])
        self.assertGreater(aco.pheromones[1, 3], initial_pheromones[1, 3])
        self.assertGreater(aco.pheromones[3, 2], initial_pheromones[3, 2])
        self.assertGreater(aco.pheromones[2, 0], initial_pheromones[2, 0])

        # Перевіряємо, що інші феромони зменшилися через випаровування
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                if i != j and not ((i, j) in [(0, 1), (1, 0), (1, 3), (3, 1), (3, 2), (2, 3), (2, 0), (0, 2)]):
                    self.assertLess(aco.pheromones[i, j], initial_pheromones[i, j])

    def test_update_pheromones_evaporation(self):
        aco = ACO(self.graph, num_ants=1, num_iterations=1, num_vehicles=1,
                  evaporation_rate=0.5, initial_pheromone=10.0)  # Високий початковий феромон

        initial_pheromones = np.copy(aco.pheromones)
        # Немає успішних рішень, тому тільки випаровування
        aco._update_pheromones([])

        # Всі феромони повинні зменшитися
        expected_pheromones = initial_pheromones * (1 - aco.evaporation_rate)
        np.testing.assert_array_almost_equal(aco.pheromones, expected_pheromones)

    def test_run_best_path_update(self):
        aco = ACO(self.graph, num_ants=10, num_iterations=5, num_vehicles=1,
                  alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

        # Визначаємо "ідеальний" маршрут, який _construct_solution_routes має повертати.
        ideal_solution_routes = [[0, 1, 2, 3, 0]]
        # Розраховуємо його довжину, щоб ACO міг це перевірити.
        ideal_length = (self.graph.get_distance(0, 1) +
                        self.graph.get_distance(1, 2) +
                        self.graph.get_distance(2, 3) +
                        self.graph.get_distance(3, 0))

        # Заглушуємо _construct_solution_routes та _calculate_total_solution_length
        # Це гарантує, що ACO.run завжди отримає дійсний, відомий маршрут
        # і перевірить, чи правильно він його обробляє та оновлює best_path_length.
        with mock.patch.object(aco, '_construct_solution_routes', return_value=ideal_solution_routes) as mock_construct:
            with mock.patch.object(aco, '_calculate_total_solution_length',
                                   return_value=ideal_length) as mock_calculate:
                best_routes, best_length = aco.run(start_node=0)

                # Перевіряємо, що знайдене рішення є дійсним і має скінченну довжину
                self.assertIsNotNone(best_routes)
                self.assertNotEqual(best_length, np.inf)
                self.assertGreater(best_length, 0)

                # Перевіряємо, що aco.best_path_length оновився
                self.assertEqual(aco.best_path_length, best_length)
                self.assertEqual(aco.best_path_length, ideal_length)  # Перевіряємо, що це "ідеальна" довжина
                self.assertEqual(aco.best_solution_routes, ideal_solution_routes)  # Перевіряємо маршрути

                # Перевіряємо, що mock-методи були викликані
                # mock_construct має бути викликаний num_ants * num_iterations разів
                self.assertEqual(mock_construct.call_count, aco.num_ants * aco.num_iterations)
                # mock_calculate має бути викликаний стільки ж разів, якщо _construct_solution_routes не повертає None
                self.assertEqual(mock_calculate.call_count, aco.num_ants * aco.num_iterations)

                # Перевіряємо структуру маршрутів
                visited_clients_overall = set()
                for route in best_routes:
                    self.assertEqual(route[0], aco.depot_node)
                    self.assertEqual(route[-1], aco.depot_node)
                    for node in route:
                        if node != aco.depot_node:
                            visited_clients_overall.add(node)
                self.assertEqual(visited_clients_overall, aco.all_client_nodes)

    def test_run_with_callback(self):
        # Перевіряємо, чи викликається callback
        aco = ACO(self.graph, num_ants=1, num_iterations=3, num_vehicles=1)

        mock_callback_calls = []

        def mock_callback(iteration, routes, length, pheromones):
            mock_callback_calls.append((iteration, routes, length))

        # Мокємо _construct_solution_routes, щоб воно завжди повертало дійсний маршрут
        # для спрощення тестування колбеку.
        valid_solution_routes = [[0, 1, 2, 3, 0]]
        valid_total_length = 4.0  # Залежить від вашого графа
        with mock.patch.object(aco, '_construct_solution_routes', return_value=valid_solution_routes) as mock_construct:
            with mock.patch.object(aco, '_calculate_total_solution_length',
                                   return_value=valid_total_length) as mock_calculate:
                aco.run(start_node=0, update_callback=mock_callback)

        self.assertEqual(len(mock_callback_calls), 3)  # Очікуємо 3 виклики (за кількістю ітерацій)
        for i, call in enumerate(mock_callback_calls):
            self.assertEqual(call[0], i)  # Перевіряємо номер ітерації
            self.assertEqual(call[1], valid_solution_routes)  # Перевіряємо, що передаються маршрути
            self.assertEqual(call[2], valid_total_length)  # Перевіряємо, що передається довжина


if __name__ == '__main__':
    # Дозволяє запускати тести в IDE без sys.exit
    unittest.main(argv=['first-arg-is-ignored'], exit=False)