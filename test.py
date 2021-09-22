import redis
import os
import time

pid = os.getpid()
time.sleep(10)

r = redis.Redis()

print(r.set("test_"+str(pid),pid))
print(r.get("test_"+str(pid)))

time.sleep(20)