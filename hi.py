import time

start_time = time.time()

a = 0
for r in range(1000):
      for c in range(1000):
         a += 1

elapsed_time = time.time() - start_time

print 'time cost = ',elapsed_time, a
