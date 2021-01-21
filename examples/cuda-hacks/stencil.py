from devito import *

configuration['log-level'] = 'PERF'

grid = Grid(shape=(80, 80, 80))

u = TimeFunction(name='u', grid=grid)

op = Operator(Eq(u.forward, u + 1.), platform='nvidiaX', language='openacc')

op.apply(time_M=100000)

print('u[0,4,4,4]=', u.data[0, 4, 4 ,4])
