from random import randint, random
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# from math import log
from multiprocessing import Pool


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
        i = randint(0, n - 1)
        x1[i] = 1 - x1[i]
        # bits_changed = 0
        # for i in range(n):
        #     if random() * n < 1:
        #         x1[i] = 1 - x1[i]
        #         bits_changed += 1
        return x1

    def log_improvement(self, logfile, f1x, f1x1):
        logfile.write('{}\n{}\n'.format(f1x, f1x1))
        logfile.write('replaced level {} with mutation of level {}\n'.format(f1x1, f1x))
        logfile.write('Hamming distances: \n')
        for i in range(len(self.pop) - 1):
            logfile.write('{}, '.format(sum((self.pop[i][j] - self.pop[i + 1][j]) ** 2 for j in range(len(self.pop) - 1))))
        logfile.write('\n')

        if (len(self.pop) + 1) // 2 + 1 in self.frequencies:
            hot_position = self.frequencies.index((len(self.pop) + 1) // 2 + 1)
            cold_position = self.frequencies.index((len(self.pop) + 1) // 2 - 1)
            if hot_position > cold_position:
                logfile.write(' ' * cold_position + ' C ' + ' ' * (hot_position - cold_position) + 'H\n')
            else:
                logfile.write(' ' * hot_position + ' H ' + ' ' * (cold_position - hot_position) + 'C\n')
            for x in self.pop:
                for i in range(len(x)):
                    if i in (cold_position, hot_position):
                        logfile.write('|')
                    logfile.write('{}'.format(x[i]))
                    if i in (cold_position, hot_position):
                        logfile.write('|')
                logfile.write('\n')
        else:
            for x in self.pop:
                for i in range(len(x)):
                    logfile.write('{}'.format(x[i]))
                logfile.write('\n')

    def iteration(self, logfile):
        f1x = randint(0, len(self.pop) - 1)
        x = self.pop[f1x]
        x1 = self.mutate(x)
        f1x1 = self.f1(x1)
        # logfile.write("{} {} {}\n".format(f1x, f1x1, bits_changed))
        ### compare frequencies if we insert x1 into population
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
            ### Uncomment if want to log the frequency vector at each iteration
            # for i in sorted(self.frequencies, reverse=True):
            #     logfile.write("{} ".format(i))
        else:
            # the way we go when we maximize sum of Hamming distances
            ham_old = sum(i * (len(self.pop) - i) for i in self.frequencies)
            ham_new = sum(i * (len(self.pop) - i) for i in new_frequencies)
            if ham_old <= ham_new:
                self.pop[f1x1] = x1
                self.frequencies = new_frequencies

        ## Uncomment if want to log the frequency vector at each iteration
        #     for i in self.frequencies:
        #         logfile.write("{} ".format(i))
        # logfile.write("\n")
        # logfile.flush()

    def run(self, logfile):
        # hot_position = self.frequencies.index((len(self.pop) + 1) // 2 + 1)
        # cold_position = self.frequencies.index((len(self.pop) + 1) // 2 - 1)
        # if hot_position > cold_position:
        #     f.write(' ' * cold_position + ' C ' + ' ' * (hot_position - cold_position) + 'H\n')
        # else:
        #     f.write(' ' * hot_position + ' H ' + ' ' * (cold_position - hot_position) + 'C\n')
        # for x in self.pop:
        #     for i in range(len(x)):
        #         if i in (cold_position, hot_position):
        #             logfile.write('|')
        #         logfile.write('{}'.format(x[i]))
        #         if i in (cold_position, hot_position):
        #             logfile.write('|')
        #     logfile.write('\n')
        i = 0
        while sum(i * (len(self.pop) - i) for i in self.frequencies) < (len(self.pop) ** 2 / 4) * (len(self.pop) - 1):
            if i >= (len(self.pop) - 1) ** 4:
                logfile.write("Timelimit\n")
                for x in self.pop:
                    for j in x:
                        logfile.write('{}'.format(j))
                    logfile.write('\n')
                logfile.flush()
                return i
            self.iteration(logfile)
            i += 1
        logfile.write("Finished: {} iterations\n".format(i))
        logfile.flush()
        return i


def complicated_population(n): # n must be odd
    pop = [[0] * n, [1] + [0] * (n - 1)]
    last_one_position = 1
    for i in range(2, (n + 1) // 2):
        first_one_position = (last_one_position + 1) % n
        last_one_position = first_one_position + i - 1
        if last_one_position < n:
            pop.append([0] * first_one_position + [1] * i + [0] * (n - last_one_position - 1))
        else:
            last_one_position = last_one_position % n
            pop.append([1] * (last_one_position + 1) + [0] * (n - i) + [1] * (n - first_one_position))
    for i in range((n + 1) // 2):
        pop.append(list(map(lambda x: 1 - x, pop[(n - 1) // 2 - i])))
    # Now we have a perfectly balanced population
    # We disturb it in level n-2
    pop[-2][0], pop[-2][1] = 1, 0
    # Now we move "hot" and "cold" columns to somewhere else by changing the bits in the middle-fitness individuals
    i = (n - 1) // 2
    if last_one_position == 0:
        pop[i][0], pop[i][1] = 0, 1
        pop[i][first_one_position - 1], pop[i][first_one_position] = 1, 0
    elif first_one_position == 1:
        pop[i + 1][0], pop[i + 1][1] = 0, 1
        pop[i + 1][last_one_position], pop[i + 1][last_one_position + 1] = 1, 0
    elif pop[i][0] == 1:
        pop[i][0], pop[i][first_one_position - 2] = 0, 1
        pop[i + 1][1], pop[i + 1][first_one_position - 1] = 1, 0
    else:
        pop[i][1], pop[i][last_one_position] = 1, 0
        pop[i + 1][0], pop[i + 1][last_one_position - 1] = 0, 1
    return pop

runs = 10

# with open('easy-log.txt', 'w') as f:
#     print(sum(GSEMO(complicated_population(11)).run(f) for i in range(runs)) / runs)
#
# exit(0)

def run_thread(thread_number):
    with open("hamming-timelimit-{}.log".format(thread_number), 'w') as f:
        for n in [2 ** i + 1 for i in range(4, 11)]:
            f.write("n = {}\n".format(n))
            for i in range(runs):
                pop = [[1] * i + [0] * (n - i) for i in range(n + 1)]
                f.write("Run {}\n".format(i))
                GSEMO(pop).run(f)

    with open("frequencies-timelimit-{}.log".format(thread_number), 'w') as f:
        for n in [2 ** i + 1 for i in range(4, 11)]:
            f.write("n = {}\n".format(n))
            for i in range(runs):
                pop = [[1] * i + [0] * (n - i) for i in range(n + 1)]
                f.write("Run {}\n".format(i))
                GSEMO(pop, 'Frequencies').run(f)

with Pool(8) as pool:
    pool.map(run_thread, list(range(1, 9)))

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