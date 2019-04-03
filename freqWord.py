from collections import Counter
import operator
import codecs
import numpy as np

f = open('/home/usuaris/veu/cescola/interlingua-fairseq/data-bin/wmt17.tokenized.16k.tr-en/train.bpe.tr')
lines = f.readlines()
print("File loaded")
words = []
lengths = []
for l in lines:
    words_in = l.split()
    words += words_in
    lengths.append(len(words_in))

print("Sentences to words")

print('Count of sentence lengths')
print(Counter(lengths))
print('Max lenght:', np.max(lengths))
print('Min length:', np.min(lengths))
print('Length mean:', np.mean(lengths))

c = dict(Counter(words))
print("Count frequencies")

sorted_c = sorted(c.items(), key=operator.itemgetter(1), reverse=True)

print("Sorting")

total = sum([v for k,v in sorted_c])

print('Total number or words', total)
print('Different words', len(sorted_c))


def printStep(c,i,perc):
    cs = [c[j][0] for j in range(0,i)]
    print("Words until ", perc, "%", len(cs))

ac = 0
milestones = [False,False,False,False,False]
for i,cc in enumerate(sorted_c):
    k,v = cc
    ac += v
    
    if float(ac)/total > 0.5 and float(ac)/total < 0.75 and not milestones[0]:
        milestones[0] = True
        printStep(sorted_c,i,50)
    if float(ac)/total >=0.75 and float(ac)/total < 0.90 and not milestones[1]:
        milestones[1] = True
        printStep(sorted_c,i,75)
    if float(ac)/total >=0.90 and float(ac)/total < 0.95 and not milestones[2]:
        milestones[2] = True
        printStep(sorted_c,i,90)
    if float(ac)/total >=0.95 and float(ac)/total < 0.99 and not milestones[3]:
        milestones[3] = True
        printStep(sorted_c,i,95)
    if float(ac)/total >=0.99  and not milestones[4]:
        milestones[4] = True
        printStep(sorted_c,i,99)
    if float(ac)/total == 1.0  and  milestones[4]:
        printStep(sorted_c,i,100)
    
        




