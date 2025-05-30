import tkinter as tk
from tkinter import ttk
import time
from graph import Graph
from aco import ACO
from visualization import AntColonyVisualizer
import threading
import random
import numpy as np
import matplotlib.pyplot as plt


class ACOApp:
    def __init__(self, master):
        self.master = master
        master.title("Мурашиний Алгоритм Оптимізації Маршрутів (VRP)")

        self.graph = None
        self.aco = None
        self.visualizer = None
        self.simulation_thread = None
        self.is_running = False

        self.create_widgets()

        plt.close('all')

    def create_widgets(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        graph_frame = ttk.LabelFrame(settings_frame, text="Налаштування Графа")
        graph_frame.pack(fill=tk.X, pady=5)

        ttk.Label(graph_frame, text="Кількість міст:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.num_nodes_var = tk.IntVar(value=10)
        ttk.Entry(graph_frame, textvariable=self.num_nodes_var, width=10).grid(row=0, column=1, padx=5, pady=2,
                                                                               sticky="ew")

        self.random_nodes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(graph_frame, text="Випадкові позиції міст", variable=self.random_nodes_var,
                        command=self.toggle_manual_input).grid(row=1, column=0, columnspan=2, padx=5, pady=2,
                                                               sticky="w")

        self.manual_positions_text = tk.Text(graph_frame, height=5, width=30, state=tk.DISABLED)
        self.manual_positions_text.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="ew")
        self.manual_positions_text.insert(tk.END, "Введіть (x,y) кожне на новому рядку, напр.\n0,0\n1,2")

        ttk.Button(graph_frame, text="Згенерувати Граф", command=self.generate_graph).grid(row=3, column=0,
                                                                                           columnspan=2, pady=5)

        aco_frame = ttk.LabelFrame(settings_frame, text="Налаштування ACO")
        aco_frame.pack(fill=tk.X, pady=5)

        ttk.Label(aco_frame, text="Кількість агентів (мурах):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.num_ants_var = tk.IntVar(value=10)
        ttk.Entry(aco_frame, textvariable=self.num_ants_var, width=10).grid(row=0, column=1, padx=5, pady=2,
                                                                            sticky="ew")

        ttk.Label(aco_frame, text="Кількість ітерацій:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.num_iterations_var = tk.IntVar(value=50)
        ttk.Entry(aco_frame, textvariable=self.num_iterations_var, width=10).grid(row=1, column=1, padx=5, pady=2,
                                                                                  sticky="ew")

        ttk.Label(aco_frame, text="Кількість транспортних засобів:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.num_vehicles_var = tk.IntVar(value=1)
        ttk.Entry(aco_frame, textvariable=self.num_vehicles_var, width=10).grid(row=2, column=1, padx=5, pady=2,
                                                                                sticky="ew")

        ttk.Label(aco_frame, text="Вага феромонів (Alpha):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.alpha_var = tk.DoubleVar(value=1.0)
        ttk.Entry(aco_frame, textvariable=self.alpha_var, width=10).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(aco_frame, text="Вага відстані (Beta):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.beta_var = tk.DoubleVar(value=2.0)
        ttk.Entry(aco_frame, textvariable=self.beta_var, width=10).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(aco_frame, text="Швидкість випаровування феромонів:").grid(row=5, column=0, padx=5, pady=2,
                                                                             sticky="w")
        self.evaporation_var = tk.DoubleVar(value=0.5)
        ttk.Entry(aco_frame, textvariable=self.evaporation_var, width=10).grid(row=5, column=1, padx=5, pady=2,
                                                                               sticky="ew")

        ttk.Label(aco_frame, text="Коефіцієнт відкладення феромонів (Q):").grid(row=6, column=0, padx=5, pady=2,
                                                                                sticky="w")
        self.q_var = tk.DoubleVar(value=100)
        ttk.Entry(aco_frame, textvariable=self.q_var, width=10).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        control_frame = ttk.LabelFrame(settings_frame, text="Управління Симуляцією")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Швидкість анімації (мс):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.animation_speed_var = tk.IntVar(value=100)
        ttk.Scale(control_frame, from_=10, to=1000, orient="horizontal",
                  variable=self.animation_speed_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        self.start_button = ttk.Button(control_frame, text="Запустити ACO", command=self.start_simulation)
        self.start_button.grid(row=1, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Зупинити ACO", command=self.stop_simulation,
                                      state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=5, pady=5)

        self.status_label = ttk.Label(self.master, text="Очікування налаштувань...")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.viz_frame = ttk.LabelFrame(main_frame, text="Візуалізація")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def toggle_manual_input(self):
        if self.random_nodes_var.get():
            self.manual_positions_text.config(state=tk.DISABLED)
        else:
            self.manual_positions_text.config(state=tk.NORMAL)

    def generate_graph(self):
        if self.visualizer:
            plt.close(self.visualizer.fig)
            self.visualizer = None
            for widget in self.viz_frame.winfo_children():
                widget.destroy()

        try:
            num_nodes = self.num_nodes_var.get()
            num_vehicles = self.num_vehicles_var.get()

            if num_nodes <= 1:
                self.status_label.config(text="Кількість міст повинна бути більше 1 (депо + мінімум 1 клієнт).", foreground="red")
                return

            if not (1 <= num_vehicles <= num_nodes - 1):
                self.status_label.config(
                    text=f"Кількість транспортних засобів має бути від 1 до {num_nodes - 1} (кількість клієнтів).",
                    foreground="red")
                return

            if self.random_nodes_var.get():
                self.graph = Graph.generate_random_graph(num_nodes)
            else:
                positions_str = self.manual_positions_text.get("1.0", tk.END).strip()
                positions = []
                for line in positions_str.split('\n'):
                    if line.strip():
                        try:
                            x, y = map(float, line.split(','))
                            positions.append((x, y))
                        except ValueError:
                            self.status_label.config(
                                text=f"Помилка формату координат: '{line}'. Введіть (x,y).", foreground="red")
                            return

                if len(positions) != num_nodes:
                    self.status_label.config(
                        text=f"Помилка: Кількість введених позицій ({len(positions)}) не відповідає кількості міст ({num_nodes}).",
                        foreground="red")
                    return

                self.graph = Graph(num_nodes)
                self.graph.set_node_positions(positions)
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                        self.graph.add_edge(i, j, dist)

            self.aco = ACO(self.graph,
                           num_ants=self.num_ants_var.get(),
                           num_iterations=self.num_iterations_var.get(),
                           num_vehicles=num_vehicles,
                           alpha=self.alpha_var.get(),
                           beta=self.beta_var.get(),
                           evaporation_rate=self.evaporation_var.get(),
                           Q=self.q_var.get())

            self.visualizer = AntColonyVisualizer(self.graph, self.viz_frame)
            self.visualizer._draw_graph(self.aco.pheromones)
            self.visualizer.canvas.draw_idle()

            self.status_label.config(text="Граф згенеровано. Готово до запуску.", foreground="green")
            self.start_button.config(state=tk.NORMAL)
        except Exception as e:
            self.status_label.config(text=f"Помилка генерації графа: {e}", foreground="red")
            print(f"Помилка генерації графа: {e}") # Для дебагу у консолі

    def start_simulation(self):
        if self.graph is None:
            self.status_label.config(text="Будь ласка, спочатку згенеруйте граф.", foreground="red")
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Симуляція запущена...", foreground="blue")

        self.simulation_thread = threading.Thread(target=self._run_aco_in_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_simulation(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Симуляція зупинена.", foreground="orange")

    def _run_aco_in_thread(self):
        try:
            def update_gui_callback(iteration, best_solution_routes_current, best_total_length_current,
                                    pheromones_current):
                if self.is_running:
                    self.master.after(self.animation_speed_var.get(),
                                      lambda: self.visualizer.update_plot(iteration, best_solution_routes_current,
                                                                          best_total_length_current,
                                                                          pheromones_current))
                time.sleep(self.animation_speed_var.get() / 1000.0)

            self.aco.run(start_node=0, update_callback=update_gui_callback)

            self.master.after(0, lambda: self.status_label.config(
                text=f"ACO завершено! Найкраща загальна довжина: {self.aco.best_path_length:.2f}", foreground="green"))
            self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

        except Exception as e:
            self.master.after(0, lambda: self.status_label.config(text=f"Помилка ACO: {e}", foreground="red"))
            self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            print(f"Помилка в потоці ACO: {e}")
        finally:
            self.is_running = False


if __name__ == "__main__":
    root = tk.Tk()
    app = ACOApp(root)
    root.mainloop()