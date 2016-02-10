from random import gauss, random

data = ((gauss(0.0,10.0),gauss(0.0,1.0)) if random()<0.5 else \
        (gauss(0.0,1.0),gauss(0.0,10.0)) for i in range(1000))

with open('xdata.csv','w') as f:
    f.write(''.join('%f,%f\n'%d for d in data))
