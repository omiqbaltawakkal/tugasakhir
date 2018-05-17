import xlrd
import numpy
import operator
import math
import random
import matplotlib.pyplot as plt
import collections

# numpy.seterr(divide='ignore', invalid='ignore')

def opens(filename, indeks):
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(indeks)
    data = [[str(sheet.cell_value(r, c)) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
    return data

def modelling(mahasiswa):
	data = []
	for x in range(1,len(mahasiswa)):
		temp = []
		temp.append(int(float(mahasiswa[x][0])))
		temp.append(mahasiswa[x][1])
		temp.append(int(float(mahasiswa[x][2])))
		pbb = []
		pbb.append(int(float(mahasiswa[x][3])))
		if int(float(mahasiswa[x][4])) != 0:
			pbb.append(int(float(mahasiswa[x][4])))
		temp.append(pbb)
		data.append(temp)
	return data

def modellingdosen(datadosen):
	data = []
	data.append(datadosen[0])
	for x in range(1, len(datadosen)):
		temp = []
		# temp.append(x)
		for y in range(2,len(datadosen[0])):
			temp.append(int(float(datadosen[x][y])))
		data.append(temp)
	return data

def class_dosen(modeldosen):
	dosenicm =[]
	dosenside =[]
	dosentele =[]
	for x in range(len(modeldosen)):
		if modeldosen[x][0] == 1:
			dosenicm.append(x)
		elif modeldosen[x][0] == 2:
			dosenside.append(x)
		elif modeldosen[x][0] == 3:
			dosentele.append(x)
	return dosenicm, dosenside, dosentele

def mod(modelmhs):
	return numpy.array([[round(random.random(), 3) for x in range(0,4)] for y in range(0, len(modelmhs))])

def diskrit_mod(indiv):
	indi = []
	for x in range(len(indiv)):
		temp_item = []
		idx_pbb = int(round(indiv[x][0]))
		try:
			pbb = modelmhs[x][-1][idx_pbb]
		except IndexError:
			pbb = modelmhs[x][-1][0]
		temp_item.append(pbb)
		if modelmhs[x][2] == 1:
			idx_pgj1 = int(round(indiv[x][1]*len(icm)))
			idx_pgj2 = int(round(indiv[x][2]*len(icm)))
		elif modelmhs[x][2] == 2:
			idx_pgj1 = int(round(indiv[x][1]*len(side)))
			idx_pgj2 = int(round(indiv[x][2]*len(side)))
		else:
			idx_pgj1 = int(round(indiv[x][1]*len(tele)))
			idx_pgj2 = int(round(indiv[x][2]*len(tele)))
		idx_waktu = int(round(indiv[x][3]*max_time))
		temp_item.append(idx_pgj1)
		temp_item.append(idx_pgj2)
		temp_item.append(idx_waktu)
		indi.append(temp_item)
	return indi

def nilai_fitness(new_populasi):
	fitnesss = []
	for indiv in new_populasi:
		nilai = single_fitness(indiv)
		fitnesss.append(nilai)
	return fitnesss

def single_fitness(indiv):
	nilai = 0.0
	for item in indiv:
		# hard
		ex = ((item[-1]/room) % (day*3))
		for x in range(3):
			if (modeldosen[item[x]][ex] == 1):
				nilai -= 2.0
			else:
				nilai +=1.0
		#soft
		d = {}
		for item in indiv:
			for angka in item:
				if angka not in d:
					d[angka] = 1
				else:
					d[angka] +=1
		for isd in d.values():
			if isd > (1/3) * day:
				nilai -= 0.5
	return nilai

# def postition(populasi, fitnesss):
# 	pos = []
# 	half = len(populasi[0])/2
# 	for indiv in populasi:
# 		if indiv == best_individu:
# 			pos.append([0,0])
# 		else:
# 			#depan, x koordinat
# 			x = len(numpy.intersect1d(numpy.unique(indiv[:half]), numpy.unique(best_individu[:half])))   # need review
# 			#belakang, y koordinat
# 			y = len(numpy.intersect1d(numpy.unique(indiv[half:]), numpy.unique(best_individu[half:])))
# 			pos.append([x,y])
# 	return pos

def sensing_distance(populasi):
	sensing = []
	for x in range(0, len(populasi)):
		numb =0.0
		for y in range(0, len(populasi)):
			if x != y:
				numb += abs(numpy.linalg.norm(numpy.array(populasi[x]) - numpy.array(populasi[y])))
		sensing.append(numb/(5*len(populasi)))
	return sensing 

def induced(indiv, n_old, sd):
	neighbor = []
	for indiv_1 in populasi:
		if numpy.linalg.norm(indiv - indiv_1) <= sd:
			neighbor.append(indiv_1)
	# neighbor.remove(indiv)
	alpha_local = numpy.array([[0.0 for atr in item] for item in indiv])
	# alpha_local = None
	for indiv_2 in neighbor:
		x_ij = (indiv_2 - indiv) / (numpy.linalg.norm(indiv_2 - indiv) + epsilon)
		k_ij =  single_fitness(diskrit_mod(indiv)) - single_fitness(diskrit_mod(indiv_2)) / float(fitness_worst - fitness_best)
		local = x_ij*k_ij
		alpha_local += local 
	k_ibest = (single_fitness(diskrit_mod(indiv)) - fitness_best) / float(fitness_worst - fitness_best)
	x_ibest = (best_individu - indiv) / (numpy.linalg.norm(best_individu - indiv) + epsilon)
	alpha_target = c_best * k_ibest * x_ibest
	alpha_i = alpha_target + alpha_local
	return (n_max*alpha_i) + (w_n * n_old)

def foraging(indiv, food, k_iibest, x_iibest, f_old):
	#x_food replaced by food
	k_ifood = single_fitness(diskrit_mod(indiv)) - single_fitness(diskrit_mod(food)) / float(fitness_worst - fitness_best)
	x_ifood = (food - indiv) / (numpy.linalg.norm(food - indiv) + epsilon)
	b_ifood = c_food * k_ifood * x_ifood

	b_ibest = k_iibest * x_iibest
	b_i = b_ifood + b_ibest
	return v_f * b_i + w_f * f_old

def diffusion():
	# gamma = numpy.array([[round(random.uniform(-1,1), 3) for x in range(0,4)] for y in range(0, len(modelmhs))])
	gamma = numpy.array([[round(random.uniform(-1,1), 3) for atr in item] for item in indiv])
	return d_max *(1- (itter/maxs))* gamma

def tourney(populasi, tournament_size):
	selection = populasi[random.randint(0, len(populasi)-1)]
	for x in range(0, tournament_size):
		ind = populasi[random.randint(0, len(populasi)-1)]
		if single_fitness(diskrit_mod(selection)) <= single_fitness(diskrit_mod(ind)):
			selection = ind
	return selection

def crossover(parent_1, parent_2):
	crossover_point = random.randint(0, len(parent_1))
	shape = parent_1.shape
	offspring_1 = numpy.append(parent_1[:crossover_point], parent_2[crossover_point:]).reshape(shape)
	offspring_2 = numpy.append(parent_1[crossover_point:], parent_2[:crossover_point]).reshape(shape)
	return offspring_1, offspring_2

def mutation(indiv):
	if random.random() < mutate_prob:
		# indiv[random.randint(0,len(indiv)-1)][random.randint(0, len(indiv[0])-1)] += numpy.array(0.05/single_fitness(diskrit_mod(indiv)))
		indiv[random.randint(0,len(indiv)-1)][-1] += 0.005
	return indiv

def scale_linear(rawpoints, high=1.0, low=0.0):
    mins = numpy.min(rawpoints)
    maxs = numpy.max(rawpoints)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng )

def output(indiv):
	hard = 0
	soft = 0
	for item in indiv:
		# hard
		nilai = 0
		ex = ((item[-1]/room) % (day*3))
		for x in range(3):
			if (modeldosen[item[x]][ex] == 1):
				nilai += 1
		#soft
		d = {}
		for item in indiv:
			for angka in item:
				if angka not in d:
					d[angka] = 1
				else:
					d[angka] +=1
		for isd in d.values():
			if isd > (1/3) * day:
				soft +=1
	return hard, soft

if __name__ == '__main__':
	day = 5
	week = 2
	room = 5
	max_time = day * 3 * room # by day
	# max_time = week * 5 * 3 * room # by week
	# print (max_time)
	num_krill = 50
	maxs = 200
	d_max = random.uniform(0.002, 0.010)
	n_max = 0.01
	v_f = 0.002
	w_f = random.random()
	w_n = random.random()
	epsilon = 0.0001
	c_t = 1
	mutate_prob = 0.0001
	rawdatamahasiswa = opens('dummy.xlsx',0)
	rawdatadosen = opens('data-mahasiswa.xlsx',1) # [kode, nama, KK, jadwal 1 - 15]
	# datadosen= rawdatadosen[1:]
	# print (rawdatamahasiswa)
	modelmhs = modelling(rawdatamahasiswa)
	# print (modelmhs)
	modeldosen = modellingdosen(rawdatadosen) # masih dengan header
	# print modeldosen
	icm , side, tele = class_dosen(modeldosen) # pembagian dosen
	# print (icm, side, tele)
	# indiv = mod(modelmhs)
	# print (indiv)
	populasi = []
	for x in range(num_krill):
		indiv = mod(modelmhs)
		populasi.append(indiv)

	x_iibest = numpy.array([[[0.0 for atr in item] for item in indiv] for indiv in populasi])
	k_iibest = [0 for indiv in populasi]
	# print k_iibest
	old_avg =[]
	new_avg = []

	n_old = numpy.array([[[0.0 for atr in item] for item in indiv] for indiv in populasi])
	f_old = numpy.array([[[0.0 for atr in item] for item in indiv] for indiv in populasi])
	# n_old = numpy.array([])
	# f_old = numpy.array([])

	# ----------- Generation Start
	for itter in range(maxs):
		print "Generasi ", itter
	# itter = 1	
		# ----- Nilai Fitness
		# print "calculating fitness value"
		new_populasi = []
		for indiv in populasi:
			news = diskrit_mod(indiv)
			new_populasi.append(news)
		# print (new_populasi)
		fits = nilai_fitness(new_populasi)
		best_individu = populasi[fits.index(max(fits))]
		fitness_best = max(fits)
		fitness_worst = min(fits)

		old_avg.append(numpy.mean(fits))

		# ----------- Sensing Distance
		# print "calculating sensing distance"
		sd = sensing_distance(populasi)
		# print len(sd)

		c_best = 2 *(random.random() + (itter/float(maxs)))
		c_food = 2 * (1 - (itter/float(maxs)))
		food = mod(modelmhs)

		for x in range(len(populasi)):
			# ----------- Neighbor Induced
			# print "calculating neighbor induced"
			# n_old = numpy.array([[0.0 for x in range(0,4)] for y in range(0, len(modelmhs))])
			# print populasi[x]
			induced_i = induced(populasi[x],n_old[x], sd[x])
			# print "induced", type(induced_i)
			n_old[x] = induced_i
			# print len(neighbor_induced)

			# ----------- Food Attraction
			# print "calculating food Attraction"
			# f_old = numpy.array([[0.0 for x in range(0,4)] for y in range(0, len(modelmhs))])
			forage_i = foraging(populasi[x], food, k_iibest[x], x_iibest[x], f_old[x])
			x_iibest[x] = forage_i
			# f_old = forage_i
			# print "forage", type(forage_i)

			# ----------- Random Diffusion
			# print "diffusion calc"
			diff_i = diffusion()
			# diffusion_induced.append(diff_i)
			# print "diffusion", type(diff_i)

			# ----------- Motion Process
			x_i = induced_i + forage_i + diff_i
			# print "x_i ", type(x_i)
			add = x_i.size
			move = c_t * add * x_i
			temp = scale_linear(populasi[x] + move)
			populasi[x] = temp
			k_iibest[x] = single_fitness(diskrit_mod(populasi[x]))
			# print numpy.all(numpy.isnan(temp))
			# ----------- END OF MOVEMENT LOOPS

		# ----------- Crossover dan mutasi
		# print "crossing over and mutating"
		offsprings = []
		for x in range(len(populasi)/2):
			parent_1, parent_2 = tourney(populasi,2), tourney(populasi,2)
			parent_1, parent_2 = populasi[random.randint(0, len(populasi)-1)], populasi[random.randint(0, len(populasi)-1)]
			offs_1, offs_2 = crossover(parent_1,parent_2)
			offs_1, offs_2 = mutation(offs_1), mutation(offs_2)
			offsprings.extend([offs_1, offs_2])
		# print len(offsprings)
		
		# ----------- Regenerasi Populasi
		# print "changing bad individual by offsprings"
		regPopulasi = {}
		for x in range(len(offsprings)):
			regPopulasi[x] = single_fitness(diskrit_mod(offsprings[x]))
			# print offs, single_fitness(diskrit_mod(offs))
		# print regPopulasi
		idx_ready = sorted(regPopulasi, key = regPopulasi.get, reverse=True)[:len(populasi)/2]
		# regen = [offsprings[x] for x in idx_ready]
		# print idx_ready

		populasi_change = {}
		for y in range(len(fits)):
			populasi_change[y] = fits[y]
		idx_populasi = sorted(populasi_change, key=populasi_change.get, reverse=False)[:len(populasi)/2]
		# print idx_populasi
		for x, y in zip(idx_ready, idx_populasi):
			# print x,y
			populasi[y] = offsprings[x]

		new_fits = []
		for indiv in populasi:
			new_fits.append(single_fitness(diskrit_mod(indiv)))
		new_avg.append(numpy.mean(new_fits))
	
	# print "Going next Generation"

	# ------------ Output
	# print rawdatadosen[1]
	for x in range(len(populasi)):
		with open('propose jadwal-'+str(x)+'.txt', 'w') as m:
			out = diskrit_mod(populasi[x])
			for item, mhs in zip(out, modelmhs):
				dospbb = rawdatadosen[item[0]][0]
				dospgj1 = rawdatadosen[item[1]][0]
				dospgj2 = rawdatadosen[item[2]][0]
				slot_hari = item[-1]
				m.write(str(mhs[0])+', '+ str(dospbb)+', '+ str(dospgj1)+', '+str(dospgj2)+','+str(slot_hari)+'\n')
			hard, soft = output(out)
			m.write('hard constraint broke '+str(hard)+'\n'+'Soft constraint '+str(soft))

	# gens = [x for x in range(maxs)]
	# # print old_avg, new_avg, gens
	# fig = plt.figure()
	# plt.plot(gens, old_avg, c= 'green', label= "old_avg")
	# plt.plot(gens, new_avg, c = 'red', label= "new_avg")
	# plt.legend(loc='lower right')
	# plt.xlabel("Generation- ")
	# plt.ylabel("Average Fitness")
	# plt.xlim(-1, maxs)
	# plt.ylim(0, 100)
	# plt.grid()
	# plt.show()