from random import randint, random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import log
from multiprocessing import  Pool


class GSEMO:
    def __init__(self, pop, diversity_measure="Hamming"):
        self.pop = pop
        self.f1 = sum
        self.frequencies = [sum(x[j] for x in pop) for j in range(len(pop[0]))]
        self.diversity_measure = diversity_measure
        # self.log = []

    @staticmethod
    def mutate(x):
        x1 = x.copy()
        n = len(x)
        bits_changed = 0
        for i in range(n):
            if random() * n < 1:
                x1[i] = 1 - x1[i]
                bits_changed += 1
        return x1, bits_changed

    def iteration(self, logfile):
        f1x = randint(0, len(self.pop) - 1)
        x = self.pop[f1x]
        x1, bits_changed = self.mutate(x)
        f1x1 = self.f1(x1)
        logfile.write("{} {} {}\n".format(f1x, f1x1, bits_changed))
        # compare frequencies if we insert x1 into population
        new_frequencies = self.frequencies.copy()
        for i in range(len(x)):
            new_frequencies[i] += x1[i] - self.pop[f1x1][i]
        if self.diversity_measure != "Hamming":
            # the way we go when we minimize sorted frequency vectors
            nf_sorted = sorted(new_frequencies, reverse=True)
            f_sorted = sorted(self.frequencies, reverse=True)
            if nf_sorted <= f_sorted:
                self.pop[f1x1] = x1
                self.frequencies = new_frequencies
            for i in sorted(self.frequencies, reverse=True):
                logfile.write("{} ".format(i))
        else:
            # the way we go when we maximize sum of Hamming distances
            ham_old = sum(i * (len(self.pop) - i) for i in self.frequencies)
            ham_new = sum(i * (len(self.pop) - i) for i in new_frequencies)
            if ham_old <= ham_new:
                self.pop[f1x1] = x1
                self.frequencies = new_frequencies
            for i in self.frequencies:
                logfile.write("{} ".format(i))
        logfile.write("\n")
        logfile.flush()

    def run(self, logfile):
        i = 0
        while sorted(self.frequencies)[-1] * 2 > len(self.pop):
            self.iteration(logfile)
            i += 1
        logfile.write("Finished: {} iterations\n".format(i))
        logfile.flush()


runs = 2


def run_thread(thread_number):
    with open("hamming-{}.log".format(thread_number), 'w') as f:
        for n in [2 ** i + 1 for i in range(4, 6)]:
            f.write("n = {}\n".format(n))
            for i in range(runs):
                pop = [[1] * i + [0] * (n - i) for i in range(n + 1)]
                f.write("Run {}\n".format(i))
                GSEMO(pop).run(f)

with Pool(4) as pool:
    pool.map(run_thread, list(range(1, 5)))

# with open("output.log", 'r') as f:
#     runtimes = [float(s) for s in f.readline().split()]
# problem_sizes = [2 ** i + 1 for i in range(4, 9)]
# for i in range(len(runtimes)):
#     runtimes[i] = runtimes[i] / (problem_sizes[i] ** 2 * log(problem_sizes[i]))
# plt.plot(problem_sizes, runtimes, 'bo-')
# plt.show()


# animation
# fig, ax = plt.subplots()
# xdata, ydata = list(range(n)), []
# ln, = plt.plot([], [], 'ro')
#
# def init():
#     ax.set_xlim(-1, n + 1)
#     ax.set_ylim(-1, n + 2)
#     return ln,
#
# def update(frame):
#     ydata = frame
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=log, init_func=init, blit=True, interval=1)
# plt.show()