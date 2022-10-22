f1 = open("./Testcases/output/output00.txt",'r')
d1 = f1.read().split("\n")[:-1]
f1.close()

f2 = open("trainingAndTest/sample-test.out.json",'r')
d2 = f2.read().split("\n")[:-1]
f2.close()

count=0
for (i1,i2) in zip(list(map(float,d1)),list(map(float,d2))) :
    if abs(i1-i2) > 1 :
        count+=1

print((len(d1)-count)/len(d1))