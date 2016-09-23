import sys
import csv

colIdx = int(sys.argv[1])
fileName = sys.argv[2]
fhand = open(fileName)
lst = list()
for line in fhand:
    line = line.strip()
    t = line.split(' ')
    lst.append(float(t[colIdx]))

fout = open('ans1.txt', 'w')
lst.sort()
for ele in lst[:-1]:
    fout.write('%s,' % ele)
    #print '%s,' % ele,; sys.stdout.softspace = False;
fout.write(str(lst[-1]))
fout.close()

