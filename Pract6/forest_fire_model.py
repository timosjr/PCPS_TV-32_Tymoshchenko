import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Стани клітин
TREE = 0      # Незаймана
BURNING = 1   # Горить
EMPTY = 2     # Згоріла

# Параметри
GRID_SIZE = 50
P_BURN = 0.6     # Ймовірність загоряння сусіда
T_BURN = 70       # Скільки кроків горить клітина

# Ініціалізація стану сітки
state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Початковий вогонь у центрі
center = GRID_SIZE // 2
state[center, center] = BURNING
burn_time[center, center] = T_BURN


cmap = ListedColormap(["green", "orange", "red"])
colors = {
    TREE: 0,
    BURNING: 1,
    EMPTY: 2
}

# Правила переходу
def step(state, burn_time):
    new_state = state.copy()
    new_burn_time = burn_time.copy()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i, j] == TREE:
                # Перевірка сусідів
                neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        if state[ni, nj] == BURNING and np.random.rand() < P_BURN:
                            new_state[i, j] = BURNING
                            new_burn_time[i, j] = T_BURN
                            break
            elif state[i, j] == BURNING:
                new_burn_time[i, j] -= 1
                if new_burn_time[i, j] <= 0:
                    new_state[i, j] = EMPTY

    return new_state, new_burn_time

# Ініціалізація графіку
fig, ax = plt.subplots()
im = ax.imshow(np.vectorize(colors.get)(state), cmap=cmap, vmin=0, vmax=2)
ax.set_title("Модель лісової пожежі")

# Анімація
def update(frame):
    global state, burn_time
    state, burn_time = step(state, burn_time)
    im.set_data(np.vectorize(colors.get)(state))
    return [im]

ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=True)

plt.show()