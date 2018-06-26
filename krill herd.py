import xlrd
import numpy
import operator
import math
import random
import time
import copy
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
        	# for y in range(5):
            dosenicm.append(x)
        elif modeldosen[x][0] == 2:
        	# for y in range(5):
            dosenside.append(x)
        elif modeldosen[x][0] == 3:
        	# for y in range(5):
            dosentele.append(x)
    return dosenicm, dosenside, dosentele

def mod(modelmhs):
    return numpy.array([[random.random() for x in range(0,4)] for y in range(0, len(modelmhs))])

def diskrit_mod(indiv):
    indi = []
    max_times = [x for x in range(max_time)] 
    for x in range(len(indiv)):
        temp_item = []
        idx_pbb = int(round(indiv[x][0]))
        try:
            pbb = modelmhs[x][-1][idx_pbb]
        except IndexError:
            pbb = modelmhs[x][-1][0]
        temp_item.append(pbb)
        if modelmhs[x][2] == 1:
            dosen_icm = copy.copy(icm)
            # dosen_icm.remove(pbb)
            # for item in modelmhs[x][-1]:
            	# while item in dosen_icm:
            		# dosen_icm.remove(item)
            [dosen_icm.remove(item) for item in modelmhs[x][-1]]
            idx_pgj1 = int(round(indiv[x][1]*(len(dosen_icm)-1)))
            dos = dosen_icm[idx_pgj1]
            temp_item.append(dos)
            # while dos in dosen_icm:
            	# dosen_icm.remove(dos)
            dosen_icm.remove(dosen_icm[idx_pgj1])
            idx_pgj2 = int(round(indiv[x][2]*(len(dosen_icm)-1)))
            temp_item.append(dosen_icm[idx_pgj2])
        elif modelmhs[x][2] == 2:
            dosen_side = copy.copy(side)
            # dosen_side.remove(pbb)
            # for item in modelmhs[x][-1]:
            	# while item in dosen_side:
            		# dosen_side.remove(item)
            [dosen_side.remove(item) for item in modelmhs[x][-1]]
            idx_pgj1 = int(round(indiv[x][1]*(len(dosen_side)-1)))
            dos = dosen_side[idx_pgj1]
            temp_item.append(dos)
            # while dos in dosen_side:
            	# dosen_side.remove(dos)
            dosen_side.remove(dosen_side[idx_pgj1])
            idx_pgj2 = int(round(indiv[x][2]*(len(dosen_side)-1)))
            temp_item.append(dosen_side[idx_pgj2])
        elif modelmhs[x][2] == 3:
            dosen_tele = copy.copy(tele)
            # dosen_tele.remove(pbb)
            # for item in modelmhs[x][-1]:
            	# while item in dosen_tele:
            		# dosen_tele.remove(item)
            [dosen_tele.remove(item) for item in modelmhs[x][-1]]
            idx_pgj1 = int(round(indiv[x][1]*(len(dosen_tele)-1)))
            dos = dosen_tele[idx_pgj1]
            temp_item.append(dos)
            # while dos in dosen_tele:
            	# dosen_tele.remove(dos)
            dosen_tele.remove(dosen_tele[idx_pgj1])
            idx_pgj2 = int(round(indiv[x][2]*(len(dosen_tele)-1)))
            temp_item.append(dosen_tele[idx_pgj2])
        idx_waktu = int(round(indiv[x][3]*(len(max_times)-1)))
        # temp_item.append(idx_pgj1)
        # temp_item.append(idx_pgj2)
        temp_item.append(max_times[idx_waktu])
        max_times.remove(max_times[idx_waktu])
        indi.append(temp_item)
    return indi

def nilai_fitness(new_populasi):
    fitnesss = []
    for indiv in new_populasi:
        nilai = single_fitness(indiv)
        fitnesss.append(nilai)
    return fitnesss

def single_fitness(indiv):
    sc = 0.0
    # hard
    for item in indiv:
        exa = ((item[-1]/room) % (day*3)) % 15
        if modeldosen[item[0]][exa] == 1:
            sc +=penalty_pbb
        if modeldosen[item[1]][exa] == 1:
            sc +=penalty_pgj
        if modeldosen[item[2]][exa] == 1:
            sc +=penalty_pgj
    # soft
    # preference
    for item in indiv:
        exs = ((item[-1]/room) % (day*3)) % 15
        if reference[item[0]][exs] == 0:
            sc +=penalty_reference
        if reference[item[1]][exs] == 0:
            sc +=penalty_reference
        if reference[item[2]][exs] == 0:
            sc +=penalty_reference
    return max_penalty - sc
    # return sc

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
    return d_max *(1- (itter/float(maxs)))* gamma

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
        indiv[random.randint(0,len(indiv)-1)][-1] += 0.01
        # indiv[random.randint(0,len(indiv)-1)] += 0.1
    return indiv

def scale_linear(rawpoints, high=1.0, low=0.0):
    mins = numpy.min(rawpoints)
    maxs = numpy.max(rawpoints)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng )

def outs(indiv):
    # hard
    for item in indiv:
        exa = ((item[-1]/room) % (day*3)) % 15
        if modeldosen[item[0]][exa] == 1:
            print "bentrok"
        else:
            print "ga bentrok"
        if modeldosen[item[1]][exa] == 1:
            print "bentrok"
        else:
            print "ga bentrok"
        if modeldosen[item[2]][exa] == 1:
            print "bentrok"
        else:
            print "ga bentrok"
        print '\n'
    # soft
    # preference
    for item in indiv:
        exs = ((item[-1]/room) % (day*3)) % 15
        if reference[item[0]][exs] == 0:
            print "not ref"
        else:
            print "as wish"
        if reference[item[1]][exs] == 0:
            print "not ref"
        else:
            print "as wish"
        if reference[item[2]][exs] == 0:
            print "not ref"
        else:
            print "as wish"
        print '\n'

if __name__ == '__main__':
    start_time = time.time()
    day = 10
    week = 2
    room = 5
    max_time = day * 3 *room # by day
    # max_time = week * 5 * 3 * room # by week
    # print (max_time)
    num_krill = 50
    maxs = 500
    d_max = random.uniform(0.002, 0.010)
    n_max = 0.01
    v_f = 0.002
    w_f = random.random()
    w_n = random.random()
    epsilon = 0.001
    c_t = 0.5
    mutate_prob = 0.001
    sched = ["jumat sore", "senin pagi", "senin siang","senin sore","selasa pagi", "selasa siang","selasa sore", "rabu pagi", "rabu siang","rabu sore", "kamis pagi", "kamis siang","kamis sore", "jumat pagi", "jumat siang"]
    rawdatamahasiswa = opens('dummy.xlsx',0)
    rawdatadosen = opens('data-mahasiswa.xlsx',1) # [kode, nama, KK, jadwal 1 - 15]
    rawreference = opens('reference.xlsx',0)
    # for i, dosen in enumerate(rawdatadosen):
    # 	print i, dosen
    # datadosen= rawdatadosen[1:]
    # print (rawdatamahasiswa)
    modelmhs = modelling(rawdatamahasiswa)
    # print modelmhs
    # print modelmhs[0][-1]
    modeldosen = modellingdosen(rawdatadosen) # masih dengan header
    # for i, dosen in enumerate(modeldosen):
        # print i, dosen
    reference = modellingdosen(rawreference)
    icm , side, tele = class_dosen(modeldosen) # pembagian dosen
    # print icm
    # [side.remove(item) for item in modelmhs[0][-1]]
    # print side
    # print tele

    penalty_pbb = 5.0
    penalty_pgj = 2.0
    penalty_sched = 1.0
    penalty_reference = 0.5
    max_penalty_pbb = len(modelmhs)*penalty_pbb
    max_penalty_pgj = 2 * penalty_pgj* len(modelmhs)
    max_penalty_reference = 3*len(modelmhs)*penalty_reference
    # max_penalty = max_penalty_pbb + max_penalty_pgj
    max_penalty = max_penalty_pbb + max_penalty_pgj + max_penalty_reference
    # print max_penalty
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
    		# ----------- END OF MOVEMENT LOOPS
    
    	# ----------- Crossover dan mutasi
    	# print "crossing over and mutating"
    	offsprings = []
    	for x in range(len(populasi)/2):
    		parent_1, parent_2 = tourney(populasi,3), tourney(populasi,3)
    		parent_1, parent_2 = populasi[random.randint(0, len(populasi)-1)], populasi[random.randint(0, len(populasi)-1)]
    		offs_1, offs_2 = crossover(parent_1,parent_2)
    		offs_1, offs_2 = mutation(offs_1), mutation(offs_2)
    		offsprings.extend([offs_1, offs_2])
    	# print len(offsprings)
    
    	# ----------- Regenerasi Populasi
    	# print "changing bad individuals by offsprings"
    	regPopulasi = {}
    	for x in range(len(offsprings)):
    		regPopulasi[x] = single_fitness(diskrit_mod(offsprings[x]))
    		# print offs, single_fitness(diskrit_mod(offs))
    	# print regPopulasi
    	idx_ready = sorted(regPopulasi, key = regPopulasi.get, reverse=True)[:5]
    	# regen = [offsprings[x] for x in idx_ready]
    	# print idx_ready
    
    	populasi_change = {}
    	for y in range(len(fits)):
    		populasi_change[y] = fits[y]
    	idx_populasi = sorted(populasi_change, key=populasi_change.get, reverse=False)[:5]
    	# print idx_populasi
    	for x, y in zip(idx_ready, idx_populasi):
    		# print x,y
    		populasi[y] = offsprings[x]
    # print "Going next Generation"

    # ------------ Output
    end_time = time.time()
    new_pop = []
    for indiv in populasi:
    	naws = diskrit_mod(indiv)
    	new_pop.append(naws)
    # print (new_populasi)
    fits = nilai_fitness(new_populasi)
    best_ind = populasi[fits.index(max(fits))]

    # for x in range(len(populasi)):
    # with open('propose jadwal-'+str(num_krill)+'-krill-'+str(maxs)+'-generations--individu-ke-'+str(x)+'.txt', 'w') as m:
    with open('propose jadwal-bestindiv-'+str(num_krill)+'-krill-'+str(maxs)+'-generations.txt', 'w') as m:
    # with open('checking-'+str(num_krill)+'-krill-'+str(maxs)+'-generations.txt', 'w') as m:
    	# out = diskrit_mod(populasi[x])
    	out = diskrit_mod(best_ind)
    	m.write('Fitness Value : '+str(single_fitness(out))+'\n')
    	m.write('Max Penalty : '+str(max_penalty)+'\n')
    	# m.write('Penalty Pbb : '+str(max_penalty_pbb)+', penalty pgj : '+str(max_penalty_pgj)+', penalty_reference : '+str(max_penalty_reference)+'\n')
    	for item, mhs in zip(out, modelmhs):
    		dospbb = rawdatadosen[item[0]][0]
    		dospgj1 = rawdatadosen[item[1]][0]
    		dospgj2 = rawdatadosen[item[2]][0]
    		slot_waktu = ((item[-1]/room) % (day*3)) % 15
    		slot_ruang = ((item[-1] % (day*3)) % 15) % room
    		m.write(str(mhs[0])+', '+ str(dospbb)+', '+ str(dospgj1)+', '+str(dospgj2)+', '+str(item[-1])+', '+str(sched[slot_waktu])+', ruang '+str(slot_ruang)+'\n')
    	# m.write(str(outs(out)))

    with open('detail-running-time.txt','a') as r:
    	out = diskrit_mod(best_ind)
    	r.write('Detail:'+'\n')
    	r.write('Time Lapse : '+str(end_time-start_time)+'\n')
    	r.write('Nilai Fitness : '+str(single_fitness(out))+'\n')
    	r.write('Penalty Pbb : '+str(max_penalty_pbb)+', penalty pgj : '+str(max_penalty_pgj)+', penalty_reference : '+str(max_penalty_reference)+'\n')
    	r.write('Max Penalty : '+str(max_penalty)+'\n')
    	r.write('Percentage pbb: '+str(max_penalty_pbb/float(max_penalty))+', pgj : '+str(max_penalty_pgj/float(max_penalty))+', reference : '+str(max_penalty_reference/float(max_penalty))+'\n')
    	r.write('Percentage: '+str(single_fitness(out)/float(max_penalty))+'\n')
    	r.write('Max Generations : '+str(maxs)+'\n')
    	r.write('Num Krill : '+str(num_krill)+'\n')
    	r.write('\n')