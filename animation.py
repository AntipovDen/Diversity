from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_frequencies(n=65):
    with open("log.txt", 'r') as f:
        log = [[int(i) for i in s.split()] for s in f.readlines()]

    fig, ax = plt.subplots()
    xdata, ydata = list(range(n)), []
    frequencies, = ax.plot(list(range(n)), list(range(n)), 'bo-')

    def init():
        ax.set_xlim(-1, n + 1)
        ax.set_ylim(-1, n + 2)
        return frequencies,

    def update(frame):
        ydata = frame
        frequencies.set_data(xdata, ydata)
        return frequencies,

    ani = FuncAnimation(fig, update, frames=log, init_func=init, blit=True, interval=1)
    plt.show()


def animate_waves(n=33):
    with open("waves.txt", 'r') as f:
        log = [[int(i) for i in s.split()] for s in f.readlines()]

    fig, ax = plt.subplots(nrows=2)
    x_dist_data = list(range(n))
    distances, = ax[0].plot(list(range(n)), list(range(n)), 'ro-')

    x_div_data = []
    y_div_data = []
    diversity, = ax[1].semilogy(list(range(n)), list(range(n)), 'b-')

    current_time = 0
    time_window = n ** 2

    max_diversity = n * (n + 1) ** 2 // 4
    min_diversity = n * (n + 1) * (n + 2) // 6

    def init():
        ax[0].set_xlim(-1, n + 1)
        ax[0].set_ylim(-1, n + 2)
        ax[1].set_xlim(-1, time_window + 10)
        ax[1].set_ylim(1, max_diversity - min_diversity)
        return distances, diversity

    def update(frame):
        distances.set_data(x_dist_data, frame[1:])

        nonlocal current_time, x_div_data, y_div_data
        current_time += 1
        x_div_data.append(current_time)
        y_div_data.append(max_diversity - frame[0])
        if current_time > time_window:
            x_div_data = x_div_data[1:]
            y_div_data = y_div_data[1:]
            ax[1].set_xlim(current_time - time_window - 1, current_time + 10)
        diversity.set_data(x_div_data, y_div_data)

        return distances, diversity

    ani = FuncAnimation(fig, update, frames=log, init_func=init, blit=True, interval=10)
    plt.show()

animate_waves(63)