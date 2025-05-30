import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk


class AntColonyVisualizer:
    def __init__(self, graph, master_frame, depot_node=0):
        self.graph = graph
        self.master_frame = master_frame
        self.depot_node = depot_node

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Мурашиний Алгоритм Оптимізації Маршрутів (VRP)")
        self.ax.set_xlabel("X-координата")
        self.ax.set_ylabel("Y-координата")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master_frame)
        self.toolbar.update()
        self.canvas_widget.pack_forget()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.node_scatter = None
        self.edge_lines = {}
        self.best_path_line_segments = []
        self.status_text_obj = None

    def _draw_graph(self, pheromones):
        if self.node_scatter is None:
            xs = [pos[0] for pos in self.graph.node_positions]
            ys = [pos[1] for pos in self.graph.node_positions]
            self.node_scatter = self.ax.scatter(xs, ys, s=100, zorder=5)

            for i, (x, y) in enumerate(self.graph.node_positions):
                self.ax.text(x + 0.1, y + 0.1, str(i), fontsize=10)
                if i == self.depot_node:
                    self.ax.scatter(x, y, s=200, zorder=6, marker='*', color='purple', edgecolors='black', label='Депо')

            for i in range(self.graph.num_nodes):
                for j in range(i + 1, self.graph.num_nodes):
                    x1, y1 = self.graph.node_positions[i]
                    x2, y2 = self.graph.node_positions[j]

                    line, = self.ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.1, linewidth=1, zorder=1)
                    self.edge_lines[(i, j)] = line
                    self.edge_lines[(j, i)] = line

            self.ax.set_xlim(min(xs) - 1, max(xs) + 1)
            self.ax.set_ylim(min(ys) - 1, max(ys) + 1)

            self.status_text_obj = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                                verticalalignment='top', fontsize=12,
                                                bbox=dict(facecolor='white', alpha=0.7))

        if self.node_scatter is not None:
            max_pheromone = np.max(pheromones) if np.max(pheromones) > 0 else 1
            min_pheromone = np.min(pheromones[np.nonzero(pheromones)]) if np.min(
                pheromones[np.nonzero(pheromones)]) > 0 else 0.01

            for (u, v), line in self.edge_lines.items():
                current_pheromone = pheromones[u, v]
                if max_pheromone - min_pheromone + 1e-9 > 0:
                    line_alpha = (current_pheromone - min_pheromone) / (max_pheromone - min_pheromone + 1e-9)
                else:
                    line_alpha = 0.5
                line_alpha = max(0.1, min(line_alpha, 1.0))

                line_width = 1 + line_alpha * 3

                line.set_alpha(line_alpha)
                line.set_linewidth(line_width)
                line.set_color('gray')

    def update_plot(self, iteration, best_solution_routes, best_total_length, pheromones):
        self._draw_graph(pheromones)

        for segment in self.best_path_line_segments:
            segment.remove()
        self.best_path_line_segments.clear()

        # Малюємо найкраще рішення (кілька маршрутів)
        if best_solution_routes is not None and best_total_length != np.inf:
            # Важливо: якщо best_solution_routes є None або порожнім, то кольорів не буде.
            # Якщо best_solution_routes містить лише [0,0] маршрути, ми все одно отримаємо кольори,
            # але ці маршрути не будуть намальовані.
            num_actual_routes = len(best_solution_routes) if best_solution_routes else 1
            colors = plt.cm.get_cmap('gist_rainbow', num_actual_routes)

            if best_solution_routes:  # Перевірка на None
                for route_idx, route in enumerate(best_solution_routes):
                    # Малюємо маршрути, які не є порожніми [депо, депо]
                    if len(route) > 2 or (
                            len(route) == 2 and route[0] != route[1]):  # (депо, клієнт, депо) або (депо, клієнт)
                        for i in range(len(route) - 1):
                            node1, node2 = route[i], route[i + 1]
                            x1, y1 = self.graph.node_positions[node1]
                            x2, y2 = self.graph.node_positions[node2]

                            line, = self.ax.plot([x1, x2], [y1, y2], color=colors(route_idx), linewidth=3, zorder=2)
                            self.best_path_line_segments.append(line)

        length_str = f"{best_total_length:.2f}" if best_total_length != np.inf else "Нескінченна"
        if self.status_text_obj:
            self.status_text_obj.set_text(f"Ітерація: {iteration + 1}\nНайкраща загальна довжина: {length_str}")

        self.canvas.draw_idle()