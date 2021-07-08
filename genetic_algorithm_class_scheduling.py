import random
import matplotlib.pyplot as plt
import math
import numpy as np
import time


class GeneticAlgorithm:
	'''
	adjusting class scheduling to students preferences using genetic algorithm
	'''
	random.seed(42)

	def random_generator(self, hours_num, course_num, stud_num, population_size, preferences=True):
		'''
		generate random population and, if not defined, preferences array
		'''
		courses_list = [i for i in range(course_num)]
		self.slots_num = math.ceil(stud_num / hours_num)

		population = [[random.sample(courses_list,hours_num) for j in range(stud_num)] for i in range(population_size)]
		
		if preferences:
			preferences = [random.sample(courses_list,hours_num) for i in range(stud_num)]
			return preferences, population
		else:
			return population

	
	def fit_check(self, preferences, sub_population):
		'''
		check fit score between preferences matrix and subpopulation
		'''
		score = 0

		for i in range(len(sub_population)):
			if sub_population[i] == preferences[i]:
				score += 1
		
		return score

	
	def limit_check(self, sub_population):
		'''
		check limit of students for every class
		'''
		score = 0

		for j in range(len(sub_population[0])):
			columns = []
			for i in range(len(sub_population)):
				columns.append(sub_population[i][j])
				
			repeats = {c:columns.count(c) for c in columns}

			for value in repeats.values():
				if value > self.slots_num:
					score -= 2*int((value - self.slots_num))

		return score


	def __selection(self, preferences, population):
		'''
		genomes tournament selection for parents population
		'''

		population_size = len(population)
		subpopulation_size = len(population[0])
		parents_pop = []

		for _ in range(population_size):
			parents = []
			for j in range(subpopulation_size):
				k1 = random.randint(0, population_size-1)
				k2 = random.randint(0, population_size-1)

				parent_1 = population[k1][j]
				parent_2 = population[k2][j]

				parents_1 = parents.copy()
				parents_2 = parents.copy()

				if len(parents) > 0:
					parents_1.append(parent_1)
					parents_2.append(parent_2)
					if self.fit_check(preferences[j], parent_1) + self.limit_check(parents_1) > self.fit_check(preferences[j], parent_2) + self.limit_check(parents_2):
						parents.append(parent_1)
					else:
						parents.append(parent_2)
				else:
					if self.fit_check(preferences[j], parent_1) > self.fit_check(preferences[j], parent_2):
						parents.append(parent_1)
					else:
						parents.append(parent_2)

			parents_pop.append(parents)

		return parents_pop

	

	def __pmx(self, genome_1, genome_2):
		'''
		pmx crossover technique definition
		'''

		k1 = random.randint(0, len(genome_1)-1)
		k2 = random.randint(k1, len(genome_1)-1)
		
		genome = genome_2.copy()
		
		for i in range(k1,k2+1):
			if genome_1[i] in genome: 
				k = genome.index(genome_1[i])

				genome[k] = genome[i]
			genome[i] = genome_1[i]

		return genome

	

	def __crossover(self, parents_pop):
		'''
		crossover implementation 
		'''

		population_size = len(parents_pop)
		subpopulation_size = len(parents_pop[0])
		offsprings_pop = parents_pop.copy()

		for i in range(0, population_size, 2):
			for j in range(subpopulation_size):
				parent_1 = parents_pop[i][j]
				parent_2 = parents_pop[i+1][j]

				offsprings_pop[i][j] = self.__pmx(parent_1, parent_2)
				offsprings_pop[i][j] = self.__pmx(parent_2, parent_1)
		
		return offsprings_pop


	def __mutation(self, offsprings_pop, probability):
		'''
		random genome mutation
		'''

		population_size = len(offsprings_pop)
		subpopulation_size = len(offsprings_pop[0])

		for i in range(population_size):
			for j in range(subpopulation_size):
				u = random.uniform(0,1)
				if u < probability:
					k1 = random.randint(0, len(offsprings_pop[i][j])-1)
					k2 = random.randint(k1, len(offsprings_pop[i][j])-1)  
					offsprings_pop[i][j][k1:k2] = reversed(offsprings_pop[i][j][k1:k2])
		
		return offsprings_pop


	def repeats_count(self, population):
		'''
		count and return number of students for each class
		'''
		col_repeats = []

		for j in range(len(population[0])):
			columns = []
			for i in range(len(population)):
				columns.append(population[i][j])
				
			repeats = {c:columns.count(c) for c in columns}

			col_repeats.append(f"Column {j} repeats:\t" + str(repeats))

		return col_repeats


	def run_algorithm(self, preferences, population, n_it, probability, plot=True):
		'''
		runs genetic algorithm
		'''
		start = time.time()
		best_scores = []
		best_pops = []

		for i in range(n_it):
			print(f'Iteration {i}')
			parents_pop = self.__selection(preferences=preferences, population=population)
			offsprings_pop = self.__crossover(parents_pop=parents_pop)
			population = self.__mutation(offsprings_pop=offsprings_pop, probability=probability)
			scores = []
			for j in range(len(population)):
				score = 0
				for k in range(len(preferences)):
					score += self.fit_check(preferences[k], population[j][k]) 
				score += self.limit_check(population[j])
				scores.append(score)

			best_score = max(scores)
			best_score_index = scores.index(max(scores))
			best_pop = population[best_score_index]
			best_scores.append(best_score)
			best_pops.append(best_pop)
			
		total_best_pop = best_pops[best_scores.index(max(best_scores))]

		column_repeats = self.repeats_count(total_best_pop)

		end = time.time()
		duration = end - start

		if plot:
			fig = plt.figure()
			plt.plot(best_scores)
			fig.savefig('score_plot.png')
			plt.close()

		return max(best_scores), total_best_pop, column_repeats, duration

def main():
	ga = GeneticAlgorithm()
	n_it = 80
	probability = 0.1
	preferences, population = ga.random_generator(10,10,10,300)
	best_score, best_pop, repeats, duration = ga.run_algorithm(preferences, population, n_it, probability)

	print(np.array(best_pop),'\nBest score:', best_score, '\nDuration:', duration)

	for i in repeats:
		print (i)

if __name__ == "__main__":
    main()