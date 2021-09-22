# import redis
# import os
# import time
import pymongo
import random

# pid = os.getpid()
# time.sleep(10)

# r = redis.Redis()

# print(r.set("test_" + str(pid), pid))
# print(r.get("test_" + str(pid)))

# time.sleep(20)


# def act():


# def step():

def sortearAporte(semana, lago):
    mongo = pymongo.MongoClient("mongodb://localhost:27017/")

    db = mongo["prototipo"]
    coll = db["aportes"]

    listaAportes = list(coll.find({"semana": semana}, {"_id": 0, "semana": 0}))

    return random.choice(listaAportes)[lago]


print(sortearAporte(42))
