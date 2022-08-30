from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

n = 65

with open("log.txt", 'r') as f:
    log = [[int(i) for i in s.split()] for s in f.readlines()]

# for _ in log:
#     print(_)
# exit(0)

fig, ax = plt.subplots()
xdata, ydata = list(range(n)), []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-1, n + 2)
    return ln,

def update(frame):
    ydata = frame
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=log, init_func=init, blit=True, interval=1)
plt.show()