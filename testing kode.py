import numpy
import math
import random
from collections import Counter
from statistics import mode
import matplotlib.pyplot as plt

a = [1,1,1,1,1,2,2,2,2,2,3,4,5,6,7,8,9,10]
for item in [2,1]: 
	while item in a: 
		a.remove(item)
print a