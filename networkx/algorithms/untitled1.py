# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:52:42 2020

@author: Lampros Yfantis
"""
from itertools import count
from heapq import heappush, heappop


list1 = (3, 2, 'a')

#for i in set1:
#    print(i[1])
    
lam = {}

lam['one'] = {list1}

#for i in lam['one']:
#    print(i)
    
#lam['one'].add(1,2,3,4)

#print(lam)

lam['one'].add((3,1,'a'))
lam['one'].add((3,1,'b'))

#print(lam['one'])

lam['2'] ={(1,2)}
lam['2'].add((1,3))

#print(lam)

c=count()
ct = str(next(c))
#print(ct)
#print(type(ct))

lam[ct]={1,2,3}
#print(lam[ct])

k = (0,1,1,3,5,10)
#print(k[5])

#b = {'1': {'2': 120, '3': 128}, '14': {'2': 121, '6': 122, 'd': 10000}}
#for i,j in b.items():
#    if i == '1':
#        del[b[i]]
#print(b)

#for i, j in b.items():
#    print(j['2'])
#    
#for i in range(len(b['14'])):
#    print(i)
a = (2,2,1)
d = (1,2,2)

#print(a==d)

#for i,j in zip(a,d):
#    print(i,j)
#x = len([True for i,j in zip(a,d) if i<=j and not(a==d)])
#print(x)

print(a>d)

lampros = []


#del(lampros[2])
##print(lampros)
#
#lampros.insert(0, 's')
#
#print(lampros)        
#            
#yfa = {'u': 1}
#v = 'k'
#if v not in yfa:
#    print('ok')
#yfa['n'] = 1
#print(yfa)
#
#print(173000%86399)
#
#lam = {}
#
#lam.update({1 : {'yfa' : 3, 'kaz' : 2}})
#lam.update({2: 3})
#print(lam)
#
#for i in range(len(lampros)):
#    print(i)
#print('')   
##print(len(lampros))
#
#for i in lampros:
#    print(i)
    
push = heappush
pop = heappop

push(lampros, 5)
push(lampros, 10)
push(lampros, 2)
push(lampros, 20)
push(lampros, 3)
push(lampros, 11)

print(lampros)
tobedel = [0,1,3]

for i in range(len(tobedel)):
    del(lampros[tobedel[i]-i])
    
print(lampros)

lampros.insert(0, 1000)
print(lampros)