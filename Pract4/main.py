import numpy as np
import matplotlib.pyplot as plt

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y-x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def simulate_lorenz(initial_state, dt=0.01, num_steps=10000):
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    xs[0], ys[0], zs[0] = initial_state

    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + dt*x_dot
        ys[i + 1] = ys[i] + dt*y_dot
        zs[i + 1] = zs[i] + dt*z_dot

    return xs, ys, zs

def plot_trajectories(xs1, ys1, zs1, xs2, ys2, zs2):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(xs1, ys1, zs1, lw=0.5, color='blue', label='Початкові умови')
    ax1.set_title("Атрактор Лоренца (початкові умови)")
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(xs2, ys2, zs2, lw=0.5, color='blue', label='Змінені умови (похибка)')
    ax2.set_title("Атрактор Лоренца (з похибкою)")
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")

    plt.show()


def plot_difference(xs1, ys1, zs1, xs2, ys2, zs2, dt):
    diff = np.sqrt((xs1 - xs2) ** 2 + (ys1 - ys2) ** 2 + (zs1 - zs2) ** 2)
    time = np.arange(len(diff)) * dt

    plt.figure(figsize=(8, 4))
    plt.plot(time, diff, color='purple')
    plt.title('Відхидення між траєкторіями з часом')
    plt.xlabel('Час')
    plt.ylabel('Відстань між траєкторіями')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    dt = 0.01
    num_steps = 10000

    #Початкові умови
    init1 = (0., 1., 1.05)
    init2 = (0.01, 1., 1.05) #маленька похибка

    xs1, ys1, zs1 = simulate_lorenz(init1, dt, num_steps)
    xs2, ys2, zs2 = simulate_lorenz(init2, dt, num_steps)

    plot_trajectories(xs1, ys1, zs1, xs2, ys2, zs2)
    plot_difference(xs1, ys1, zs1, xs2, ys2, zs2, dt)