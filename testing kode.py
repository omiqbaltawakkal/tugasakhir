import numpy
import math
import random
from collections import Counter
from statistics import mode
import matplotlib.pyplot as plt

a = numpy.array([[1,2,3,4,1,12,213],[4,12,5,51,2,51,25]])

for x in range(5):
	with open('data-'+str(x)+'.txt', 'w') as f:
		f.write(str(x*2) +', ' + str(x*3) + ', ' + str(x*4))