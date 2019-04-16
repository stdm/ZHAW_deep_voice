
a = []
b = []
for i in range(100000):
    a.append(i)
    b.append(i*2)
c = zip(a,b)

idx = int(0.2 * len(c))
val, train = c[:idx], c[idx:]
len(val)
len(train)

e,f = zip(*val)
e
f
