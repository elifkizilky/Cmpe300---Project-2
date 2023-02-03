'''
Student Name: Elif Kizilkaya
Student Number: 2018400108
Compile Status: Compiling
Program Status: Working
Notes: <num_processors> should be greater than 1

	@file main.py
 	@authors Elif Kizilkaya
 			 Sena Ozpinar
 
   	@brief This code is designed to create a multiprocessor environment to calculate the
   	conditional probability of bigrams. There are n processors decided by the user.
   	One of the processor with rank 0 is the master and the others are workers with the
   	positive rank values. Master takes the input file from command line and divides it into 
   	lines and send each processor evenly shared work. Workers calculates the frequencies of 
   	bigrams and unigrams. There are 2 merging methods for the incoming data from the workers
   	to the master. If merge method MASTER is chosen, then each processor sends it data directly
   	to the master and  master merges them. If merge method WORKERS is chosen, then processors
   	send their data to the next processor and the corresponding processor merges its data with
   	the received data. The last processor sends the final data to the master. Then, master reads
   	the test file and calculates the conditional probability of the bigrams in the test file.

   	How to compile and run:
   	mpiexec -n <num_processors> -oversubscribe python3 main.py --input_file <input> --test_file <test> --merge_method <method>

   	NOTES:
   	Assumed that there is no errors in input files, test files, and command lines. Therefore, no error checks is done.
 
'''


from mpi4py import MPI
import sys

'''
	Takes 2 dictionaries as parameters and if there is a common key,
	it sums their value and merges the uncommon elements.Then returns
	the result.
'''
def merge(dict1,dict2):
	for key in dict1.keys():
		if key in dict2.keys():
			dict1[key] += dict2[key]
			del dict2[key]
	dict1.update(dict2)
	return dict1


comm = MPI.COMM_WORLD
world_size = comm.Get_size() #total number of processors	
rank = comm.Get_rank() #get the rank

input_file = "" #input file path
merge_method = "" #merge method to proceed
test_file = "" #test file path
num_workers = world_size - 1 #number of workers

result = {}

for i in range(0,len(sys.argv)): #it finds the corresponding <--parameter>, and then takes the next one as the <parameter>
	if sys.argv[i] == "--merge_method":
		merge_method = sys.argv[i+1]
	elif sys.argv[i] == "--input_file":
		input_file = sys.argv[i+1]
	elif sys.argv[i] == "--test_file":
		test_file = sys.argv[i+1]

	

if rank == 0: #master
	with open(input_file, encoding="utf-8") as inp: #since there can be Turkish characters in the text, it uses encoding="utf-8"
		text = inp.readlines() #take all the text
		lines = []
		for line in text:
			lines.append(line.strip()) #divide text into the lines

		num_lines = len(lines)
		div = num_lines // num_workers
		mod = num_lines % num_workers
		share_array = [div for i in range(0,num_workers)] #the array to hold num of lines for each worker
		
		for i in range(0,mod):
			share_array[i] += 1 #adds the remainder shares to the first processors
		
		'''
			Send evenly shared work to workers 
		'''
		begin = 0
		end = 0
		for i in range(1,num_workers+1):
			share = share_array[i-1]
			end += share
			sendList = lines[begin:end] 
			begin += share
			comm.send(sendList, dest = i, tag = 1)
			

	if merge_method == "MASTER":
		'''
		Master receives the data from the workers with tag = 2
		and then merges the received data with the previous data
		'''	
		for x in range(1, num_workers + 1):
			data = comm.recv(source = x, tag = 2)
			merge(result,data)
		
				
	elif merge_method == "WORKERS":
		'''
			Master received the data from the last processor
		'''
		result = comm.recv(source = num_workers, tag = num_workers + 1)
		

	with open(test_file, encoding="utf-8") as test: #since there can be Turkish characters in the text, it uses encoding="utf-8"
		subtext = test.readlines() #take all lines
		sentences = []
		for sentence in subtext:
			sentences.append(sentence.strip()) #divide the text into lines
		
		'''
			Calculating conditional probabilities of bigrams
		'''
		cond_probability = 0.0
		print("###########################################################")
		for sentence in sentences:
			freq1 = 0
			freq2 = 0
			if sentence in result.keys(): #if the key exists in the dictionary, take it; otherwise, frequency is 0
				freq1 = result[sentence]
		
			first_word = sentence.split()[0] #take the first word of the bigram
			first_word = first_word.strip()

			if first_word in result.keys(): #if the key exists in the dictionary, take it; otherwise, frequency is 0
				freq2 = result[first_word]
			
			if freq2 != 0: #to eliminate division by 0
				cond_probability = freq1/freq2
			else:
				cond_probability = 0
			
			print("The bigram sentence: %s\n The conditional probability: %f" % (sentence, cond_probability))
			print("###########################################################")


else: #worker

	#slave receives the data from the master with tag = 1
	data = comm.recv(source = 0, tag = 1)
	print("Worker with rank %d received number of lines: %d" % (rank, len(data)))
	
	word_dictionary = {}
	tokens = []
	for line in data:
		tokens = line.split()
		for token in tokens: #to calculate the frequency of the unigrams
			if token in word_dictionary.keys():
				word_dictionary[token] += 1
			else:
				word_dictionary[token] = 1

		bi_tokens = [tokens[i:i+2] for i in range(0, len(tokens)-1)] #divide sentence into groups of 2
		for bi_token in bi_tokens: #to calculate the frequency of the bigrams
			bigram = ' '.join([str(s) for s in bi_token]) #since list is not hashable, convert it to string
			if bigram in word_dictionary.keys():
				word_dictionary[bigram] += 1
			else:
				word_dictionary[bigram] = 1
	

	#worker sends the data to the master with tag = 2
	if merge_method == "MASTER":
		comm.send(word_dictionary, dest = 0, tag = 2)

	#worker sends the data to the next worker with the special tag between two
	elif merge_method == "WORKERS":

		if rank != 1: #if not the first processor, receive data from previous worker and merge them
			data2 = comm.recv(source = rank -1, tag = rank)
			merge(word_dictionary, data2)
			
		if rank != num_workers: #if not the last processor, send data to the next worker
			comm.send(word_dictionary, dest= rank + 1, tag = rank + 1)
		
		if rank == num_workers: #if last processor, send the data to the master
			comm.send(word_dictionary, dest = 0, tag = rank + 1)

	