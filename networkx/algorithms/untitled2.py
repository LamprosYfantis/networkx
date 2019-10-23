lam = (1, 2, 3)
print(lam[0])

people = {'123456':{'first': 'Bob', 'last':'Smith'},
          '2345343': {'first': 'Jim', 'last': 'Smith'}}

#names = list()
#first_names= list()
#last_names = list()
#for n in people.values():
#    first_names.append(n['first'])
#    last_names.append(n['last'])
#    
#print(first_names, last_names)
#
#for i, j in people.items():
#    people[i] = list(j.values())
##    people[i].update(j.values())
#    
#print(people)
#
#a=people
#print(a)
x =1
y=2

l=dict()
l.update(people)
print(l)

def change(a):
    for i, j in a.items():
        a[i] = list(j.values())
        
change(l)
print(l)
print(people)

mv = [1,2,3]
print(mv[1])