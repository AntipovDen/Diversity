log = []

with open('run.log', 'r') as f:
    for i in range(30364):
        f.readline()
        x = [int(s) for s in f.readline().split()]
        y = [int(s) for s in f.readline().split()]
        log.append([(x[j], y[j]) for j in range(len(x))])



i = 7300
while (43, 18) in log[i]:
    i -= 1
print(i)