from math import log, sqrt

from matplotlib import pyplot as plt

runtimes = dict()
for n in range(4,11):
    runtimes[2 ** n + 1] = []

for i in range(1, 9):
    with open("hamming-{}.log".format(i), 'r') as f:
        n = 0
        for s in f.readlines():
            if s[0] == 'n':
                n = int(s.split()[-1])
            elif s[0] == 'F':
                runtimes[n].append(int(s.split()[1]))


x = [i for i in runtimes.keys() if len(runtimes[i]) != 0]
y = [sum(runtimes[i]) / len(runtimes[i])  for i in x]

def median(arr):
    x = sorted(arr)
    l = len(x)
    return x[l // 4], x[l//2], x[3 * l // 4]


y1, y_med, y2 = [], [], []
for i in x:
    a, b, c = median(runtimes[i])
    y1.append(a)
    y_med.append(b)
    y2.append(c)

x_raw = []
y_raw = []
for i in x:
    for j in runtimes[i]:
        x_raw.append(i)
        y_raw.append(j)

plt.plot(x, y, 'bo-')
plt.plot(x, y_med, 'ro-')
plt.plot(x, y1, 'ro-')
plt.plot(x, y2, 'ro-')
# plt.plot(x_raw, y_raw, 'go')
plt.show()
