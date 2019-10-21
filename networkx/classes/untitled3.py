# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:59:54 2019

@author: Lampros Yfantis
"""
from random import randrange
import time
import cProfile, pstats, io
import re

def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


k ={}
for i in range(10_000_000):
    k[i]= randrange(10)



l=[]
for i in range(10_000_000):
    l.append(randrange(10))
start = time.time()    
print(l[5000000])
end = time.time()
print(end-start)


@profile
def print1():
    print(k[5000000])

@profile
def print2():
    print(l[5000000])

start = time.time()
print1()
end= time.time()
print(end-start)

start = time.time()
print2()
end= time.time()
print(end-start)


