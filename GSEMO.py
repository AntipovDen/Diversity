from random import random, randint
from multiprocessing import Pool
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation


def standard_bit_mutation(x):
    n = len(x)
    x1 = x.copy()
    for i in range(n):
        if random() * n < 1:
            x1[i] = 1 - x1[i]
    return x1


def one_bit_mutation(x):
    n = len(x)
    x1 = x.copy()
    i = randint(0, n - 1)
    x1[i] = 1 - x1[i]
    return x1


def hamming_diversity(pop):
    n = len(pop[0].s)
    diversity = 0
    for i in range(n):
        number_of_ones = sum(individual.s[i] for individual in pop)
        diversity += number_of_ones * (n + 1 - number_of_ones)
    return diversity


def hamming_opt_div(mu, n):
    return n * (mu // 2) * (mu - mu // 2)


def frequency_vector_diversity(pop):
    return sorted([sum(individual.s[i] for individual in pop) for i in range(len(pop[0]))], reverse=True)


def freq_vec_opt_div_oneminmax(mu, n):
    return [(mu + 1) // 2] * n


def better_frequency_vector_diversity(pop):
    mu = len(pop)
    frequencies = [sum(individual.s[i] for individual in pop) for i in range(len(pop[0].s))]
    return sorted([-max(frequency, mu - frequency) for frequency in frequencies])


# this is optimal for LeadingOnes+TrailingZeros and for k = 2
def opt_freq_vector(mu, n):
    res = [0] * n
    res[0] = mu - 2
    res[1] = mu - 2
    for i in range((mu - 2) // 4):
        res[2 * i + 2] = max(mu - 2 * i - 3, 2 * i + 3)
        res[2 * i + 3] = max(mu - 2 * i - 3, 2 * i + 3)
    if mu % 4 == 0:
        res[2 * ((mu - 2) // 4) + 2] = mu // 2 + 1
        res[2 * ((mu - 2) // 4) + 3] = mu // 2 + 1
        for i in range(2 * ((mu - 2) // 4) + 4, n):
            res[i] = mu // 2
    else:
        for i in range(2 * ((mu - 2) // 4) + 2, n):
            res[i] = mu // 2
    return list(map(lambda x: -x, res))


def one_min_max(x):
    i = sum(x)
    return i, len(x) - i


def leading_ones_trailing_zeros(x):
    n = len(x)
    i = 0
    while i < n and x[i] == 1:
        i += 1
    j = 0
    while j < n and x[n - j - 1] == 0:
        j += 1
    return i, j


check_feasibility_in_domination = False
feasible = lambda x: False


class Individual:
    def __init__(self, bitstring, fitness, feasible_condition=None):
        self.s = bitstring
        self.f = fitness
        self.is_feasible = feasible_condition

    def dominates(self, x):
        if check_feasibility_in_domination and feasible(self) and feasible(x):
            return False
        flag = False
        for i in range(len(self.f)):
            if self.f[i] < x.f[i]:
                return False
            elif self.f[i] > x.f[i]:
                flag = True
        return flag

    def is_dominated_by(self, x):
        return x.dominates(self)

    def is_equal(self, x):
        for i in range(len(self.s)):
            if self.s[i] != x.s[i]:
                return False
        return True


# this function computes the hypervolume to the reference point (0, 0). It assumes that the population is a pareto-front
def hypervolume(pop):
    values = [ind.f for ind in pop]
    values.sort()
    volume = (values[0][0] + 1) * (values[0][1] + 1)
    for i in range(1, len(values)):
        volume += (values[i][0] - values[i - 1][0]) * (values[i][1] + 1)
    return volume



# This algorithm maximizes the two-objective fitness function
class GSEMO:
    def __init__(self,
                 mu,
                 initial_pop,
                 fitness=one_min_max,
                 mutation=standard_bit_mutation,
                 diversity=hamming_diversity,
                 optimal_diversity=hamming_opt_div,
                 maixmize_hypervolume=False,
                 stopping_crit='front-div',
                 log_population=False,
                 return_diversity=False):
        self.mu = mu
        self.pop = initial_pop
        self.mutate = mutation
        self.div = diversity
        self.f = fitness
        self.opt_div = optimal_diversity(mu, len(self.pop[0].s))
        self.hypervolume_maximization = maixmize_hypervolume
        self.on_front_reach = None
        if stopping_crit == 'front':
            self.stop = self.stop_when_fill_the_front
        elif stopping_crit == 'div':
            self.stop = self.stop_when_optimal_diversity
        elif stopping_crit == 'front-div':
            self.stop = self.stop_when_optimal_diversity_and_front_filled
        elif 'div' in stopping_crit:
            k = int(stopping_crit.split()[1])
            self.stop = lambda: self.stop_when_optimal_diversity_and_feasible_pop(k)
            self.on_front_reach = lambda: self.stop_when_feasible_pop(k)
        else:
            self.stop = lambda: self.stop_when_feasible_pop(int(stopping_crit))
        if log_population:
            self.log_pop = []
        else:
            self.log_pop = None
        self.return_diversity = return_diversity

    # for OneMinMax to check the diversity when we fill up the front
    def stop_when_fill_the_front(self):
        return hypervolume(self.pop) >= self.mu * (self.mu + 1) // 2

    # for OneMinMax, but we stop far earlier than we fill the front (just find a diverse population)
    def stop_when_optimal_diversity(self):
        return self.div(self.pop) >= self.opt_div

    # for OneMinMax, until the optimal solution
    def stop_when_optimal_diversity_and_front_filled(self):
        return hypervolume(self.pop) >= self.mu * (self.mu + 1) // 2 and self.div(self.pop) >= self.opt_div

    # for LOTZ, till the optimal solution
    def stop_when_optimal_diversity_and_feasible_pop(self, k):
        return min([x.f[0] + x.f[1] for x in self.pop]) >= len(self.pop[0].s) - k and self.div(self.pop) >= self.opt_div

    # for LOTZ, till the population is feasible
    def stop_when_feasible_pop(self, k):
        return min([x.f[0] + x.f[1] for x in self.pop]) >= len(self.pop[0].s) - k

    def iteration(self):
        parent = self.pop[randint(0, len(self.pop) - 1)].s
        y_string = self.mutate(parent)
        y = Individual(y_string, self.f(y_string))
        # check if y is dominated by anything or is a copy of anything
        for x in self.pop:
            if x.dominates(y) or x.is_equal(y):
                return
        # remove all dominated individuals from the population and add y
        new_pop = [x for x in self.pop if not y.dominates(x)]
        new_pop.append(y)
        # remove the individual with the smallest contribution to diversity if the population is full
        if len(new_pop) > self.mu:
            # if we aim at maximizing the hypervolume, then we first must make a list of individuals which we can remove
            if self.hypervolume_maximization:
                candidates_for_removal = [0]
                max_hv = hypervolume(new_pop[1:])
                for i in range(1, self.mu + 1):
                    hv_after_removal = hypervolume(new_pop[:i] + new_pop[i + 1:])
                    if hv_after_removal == max_hv:
                        candidates_for_removal.append(i)
                    elif hv_after_removal > max_hv:
                        max_hv = hv_after_removal
                        candidates_for_removal = [i]
            else:
                candidates_for_removal = list(range(0, self.mu + 1))

            second_round_candidates = [candidates_for_removal[0]]
            max_diversity = self.div(new_pop[:candidates_for_removal[0]] + new_pop[candidates_for_removal[0] + 1:])
            for i in candidates_for_removal:
                div = self.div(new_pop[:i] + new_pop[i + 1:])
                if div == max_diversity:
                    second_round_candidates.append(i)
                elif div > max_diversity:
                    max_diversity = div
                    second_round_candidates = [i]
            remove = second_round_candidates[randint(0, len(second_round_candidates) - 1)]
            self.pop = new_pop[:remove] + new_pop[remove + 1:]
        else:
            self.pop = new_pop

    def run(self):
        t = 0
        front_reached = False
        while True:
            self.iteration()
            # logging the whole population for the purpose of animation
            if self.log_pop is not None:
                self.log_pop.append([[x.f[0] for x in self.pop], [x.f[1] for x in self.pop]])
            # logging the state when we reached the front
            if self.on_front_reach is not None and not front_reached and self.on_front_reach():
                front_reached = True
                fitnesses = [(x.f[0], x.f[1]) for x in self.pop]
                div = self.div(self.pop)
            if self.stop():
                if self.log_pop is not None:
                    return self.log_pop
                if self.return_diversity:
                    return t, self.div(self.pop)
                if self.on_front_reach is not None:
                    return t, fitnesses, div, self.div(self.pop)
                return t
            t += 1
            if t % 1000 == 0:
                print(t)
                print(self.div(self.pop))


runs = 1


def run_thread_opt_hv(thread_number):
    with open('opt_hv_{}.log'.format(thread_number), 'w') as f:
        for n in [2 ** i - 1 for i in range(4, 8)]:
            for _ in range(runs):
                initial_bitstring = [[randint(0, 1) for _ in range(n)]]
                initial_pop = [Individual(s, one_min_max(s)) for s in initial_bitstring]
                runtime, div = GSEMO(n + 1, initial_pop, maixmize_hypervolume=True, stopping_crit='front', return_diversity=True).run()
                f.write('{} {}\n'.format(runtime, div))
                f.flush()


def run_threads_opt_hv():
    with Pool(8) as pool:
        pool.map(run_thread_opt_hv, list(range(1, 9)))


def run_thread_lotz_mo(thread_number):
    global check_feasibility_in_domination, feasible
    check_feasibility_in_domination  = True
    with open('lotz_mo_{}.log'.format(thread_number), 'w') as f:
        for n in [2 ** i for i in range(3, 4)]:
            feasible = lambda x: sum(x.f) >= n - 2
            for _ in range(runs):
                initial_bitstring = [[randint(0, 1) for _ in range(n)]]
                initial_pop = [Individual(s, one_min_max(s)) for s in initial_bitstring]
                runtime, front_state, front_div, final_div = GSEMO(n, initial_pop, fitness=leading_ones_trailing_zeros, diversity=better_frequency_vector_diversity, optimal_diversity=opt_freq_vector, stopping_crit='div 2').run()
                f.write('{}\n'.format(runtime))
                f.write(' '.join(str(i[0]) for i in front_state) + '\n')
                f.write(' '.join(str(i[1]) for i in front_state) + '\n')
                f.write(' '.join(str(i) for i in front_div) + '\n')
                f.write(' '.join(str(i) for i in final_div) + '\n')
                f.flush()


run_thread_lotz_mo(0)


def animated_lotz_run():
    n = 64
    k = 2
    global feasible, check_feasibility_in_domination
    check_feasibility_in_domination = True
    feasible = lambda x: sum(x.f) >= n - k
    lotz = lambda x: leading_ones_trailing_zeros(x)
    initial_bitstring = [[randint(0, 1) for _ in range(n)]]
    initial_pop = [Individual(s, lotz(s)) for s in initial_bitstring]
    frames = GSEMO(n + 1, initial_pop, fitness=lotz, diversity=better_frequency_vector_diversity, stopping_crit='2', log_population=True).run()
    with open('run.log', 'w') as f:
        for i in range(len(frames)):
            f.write('Frame {}\n'.format(i))
            f.write(' '.join(str(s) for s in frames[i][0]))
            f.write('\n')
            f.write(' '.join(str(s) for s in frames[i][1]))
            f.write('\n')

    print(len(frames))
    fig, ax = plt.subplots()
    xdata, ydata = list(range(n)), []
    pop, = ax.plot(list(range(n)), list(range(n)), 'bo')
    ax.plot([n - k, 0], [0, n - k], 'g-')

    def init():
        ax.set_xlim(-1, n + 1)
        ax.set_ylim(-1, n + 1)
        return pop,

    t = 0
    def update(frame):
        xdata, ydata = frame
        nonlocal t
        print(t)
        t += 1
        pop.set_data(xdata, ydata)
        return pop,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=0.01)
    plt.show()


def animated_oneminmax_run():
    n = 31
    initial_bitstring = [[randint(0, 1) for _ in range(n)]]
    initial_pop = [Individual(s, one_min_max(s)) for s in initial_bitstring]
    frames = GSEMO(n + 1, initial_pop, maixmize_hypervolume=True, stopping_crit='front', log_population=True).run()
    print(len(frames))

    fig, ax = plt.subplots()
    pop, = ax.plot(list(range(n)), list(range(n)), 'bo')
    ax.plot([n, 0], [0, n], 'g-')

    def init():
        ax.set_xlim(-1, n + 1)
        ax.set_ylim(-1, n + 1)
        return pop,

    def update(frame):
        xdata, ydata = frame
        pop.set_data(xdata, ydata)
        return pop,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1)
    plt.show()


