import scipy.sparse as sp
import re
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-z", type=int)
parser.add_argument("-k", type=int)
parser.add_argument("--dir", type=str)
parser.add_argument("--doc_in", type=str)
parser.add_argument("--doc_out", type=str)
parser.add_argument("--term_in", type=str)
parser.add_argument("--term_out", type=str)
parser.add_argument("--query_in", type=str)
parser.add_argument("--query_out", type=str)
args = parser.parse_args()



pattern = re.compile(r'\W+')
print args.k



def read_single(n):
	name = args.dir + "/"
	name += str(n)
	name+='.txt'
	#print name
	f = open(name, 'r')
	text = f.read()

	f = open(name, 'r')
	title = f.readline().rstrip('\n')

	words = pattern.split(text)
	words = [element.lower() for element in words]

	bag = set(words)
	zero = [0]*(len(bag))
	dic = dict(zip(bag,zero))	
	
	for w in words:
		dic[w] = dic[w] + 1

	return [title, bag],dic


def read_data(n,kk):
	bags = []
	titles = []
	quant = []
	union_bag = set([])


	for i in range(1,n+1):
		r, foo = read_single(i)
		
		titles.append(r[0])
		bags.append(r[1])
		quant.append(foo)

	buffer_set = bags
	sz = 1;

	# divide and conquer to get union in O(nlogn)
	
	while sz<n:
		i = 0;
		while (i+sz)<n:
			buffer_set[i] = buffer_set[i].union(buffer_set[i+sz])
			i = i+sz
		sz = sz*2
	
	union_bag = list(buffer_set[0])
	union_bag.sort()

	files = list(range(n))
	words = list(range(len(union_bag)))

	print('\x1b[1;31;1m' + 'Divide and Conquer Done!' + '\x1b[1m')

	# list of dictionaries for processing


	index_titles = dict(zip(files,titles))
	titles_index = dict(zip(titles,files))

	index_words = dict(zip(words,union_bag))
	words_index = dict(zip(union_bag,words))

	for i in range(0,n):
		bags[i] = list(bags[i])

	row_idx = []
	col_idx = []
	data = []

	for i in range(0,n):
		l = len(quant[i])
		for j in quant[i]:
			row_idx.append(words_index[j]);
			col_idx.append(i)
			data.append(quant[i][j])

	M = len(union_bag)
	N = n

	A = sp.csc_matrix((data, (row_idx, col_idx)), shape=(M, N))
	
	print('\x1b[1;34;1m' + 'Sparse Matrix Done!' + '\x1b[0m')

	A = A.asfptype()
	u, s, vt = svds(A, k=kk, which = 'LM')
	v = vt.transpose()
	
	

	return N,M,kk,u,v,s,index_words, index_titles, words_index, titles_index



def my_fun():


	

	k = args.k; z = args.z

	N,M,z,u,v,s,index_words, index_titles, words_index, titles_index = read_data(5000,z)
	
	print('\x1b[1;35;1m' + 'Done SVD!' + '\x1b[0m')


	vt = v.transpose();
	ut = u.transpose();
	s_inv = 1/s;
	s_inv = np.diag(s_inv);
	ss = np.diag(s);

	vs = v*s;
	us = u*s;

	
	term_in =  args.term_in;
	term_out = args.term_out;
	doc_in = args.doc_in;
	doc_out = args.doc_out;
	query_in = args.query_in;
	query_out =  args.query_out;

	# type 1 : term searches


	f = open(term_in, 'r')
	text = f.read()
	words = pattern.split(text)
	words = [element.lower() for element in words]

	f = open(term_out, 'w')

	for x in words:
		if x!='':
			word = x.lower()
			idx = words_index[word]
			matches = []
			for i in range(0,M):
				
				foo = cosine(us[idx],us[i])
				matches.append((foo,i))

			matches.sort()

			for i in range (0,k):
				f.write(index_words[ matches[i][1]] + ';' + '\t')
			f.write('\n')

	print('\x1b[1;32;1m' + 'terms done!!!' + '\x1b[0m')


	# type 2 : documents searches

	f = open(doc_in, 'r')
	docs = []
	for line in f.readlines():
		docs.append(line)

	f = open(doc_out, 'w')

	for x in docs:
		word = x.rstrip('\n')
		idx = titles_index[word]
		matches1 = []
		for i in range(0,N):
			
			foo = cosine(vs[idx],vs[i])
			matches1.append((foo,i))

		matches1.sort()

		for i in range (0,k):
			f.write(index_titles[ matches1[i][1]] + ';' + '\t')
		f.write('\n')
	print('\x1b[1;33;1m' + 'docs done!' + '\x1b[0m')

# type 3 : querry
	f = open(query_in, 'r')
	query = []
	for line in f.readlines():
		query.append(line)

	f = open(query_out, 'w')

	for x in query:
		text = x.rstrip('\n')
		matches2 = []

		words = pattern.split(text)
		words = [element.lower() for element in words]
		bag = set(words)

		vector_words = [0]*M

		for word in bag:
			idx = words_index[word]
			vector_words[idx] = 1

		vector_words = np.array(vector_words)

		temp = np.dot (ut , vector_words)
		#vector_words = np.dot(s_inv ,temp)
		vector_words = temp
		

		for i in range(0,N):
			
			foo = cosine(vector_words,vs[i])
			matches2.append((foo,i))

		matches2.sort()

		for i in range (0,k):
			f.write(index_titles[ matches2[i][1]] + ';' + '\t')
		f.write('\n')
	print('\x1b[1;31;1m' + 'querry done!' + '\x1b[0m')


my_fun()
print('\x1b[1;32;40m' + '{:^80}'.format("Done :)")  + '\x1b[0m')