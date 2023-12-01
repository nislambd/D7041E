# import modules & set up logging
import gensim, logging, numpy as np
import help_functions as hf
import nltk
from os import path

#@author: The first version of this code is the courtesy of Vadim Selyanik

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lemmatizer = nltk.WordNetLemmatizer() # create a lemmatizer

file_path = path.abspath(__file__) # full path of your script
dir_path = path.dirname(file_path) # full path of the directory of your script

#nltk.download('wordnet', download_dir=dir_path)
nltk.download('wordnet')

sentences = []
file = open(path.join(dir_path, "lemmatized.text"), "r")

for line in file: # read the file and create list which contains all sentences found in the text
    sentences.append(line.split())
# train word2vec on the two sentences

#dimension = 50 # parameter for Word2vec size of vectors for word embedding

threshold = 0.00055 # parameter for Word2vec
print_lines = []

for dimension in [50, 100, 500, 1000]:
    for iteration in range(5):
        sum = 0.0

        # checked https://radimrehurek.com/gensim/models/word2vec.html / https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4
        model = gensim.models.Word2Vec(sentences, min_count=1, sample=threshold, sg=1,vector_size=dimension) # create model using Word2Ve with the given parameters
        #
        print(len(model.wv.key_to_index )) # check the length of the vocabulary which was formed by Word2Vec

        #The rest implements passing TOEFL tests
        i = 0 #counter for TOEFL tests
        number_of_tests = 80
        text_file = open(path.join(dir_path, 'new_toefl.txt'), 'r')
        right_answers = 0 # variable for correct answers
        number_skipped_tests = 0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
        while i < number_of_tests:
                    line = text_file.readline() #read line in the file
                    words = line.split() # extract words from the line
                    try:
                        words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                                words] # lemmatize words in the current test
                        vectors = []
                        if words[0] in model.wv: # check if there embedding for the query word
                            k = 1 #counter for loop iterating over 5 words in the test
                            vectors.append(model.wv[words[0]])
                            while k < 5:
                                if words[k] in model.wv: # if alternative has the embedding
                                    vectors.append(model.wv[words[k]]) #assing the learned vector
                                else: 
                                    vectors.append(np.random.randn(dimension)) #assing random vector
                                k += 1
                            right_answers += hf.get_answer_mod(vectors) #find the closest vector and check if it is the correct answer

                    except KeyError: # if there is no representation for the query vector than skip
                        number_skipped_tests += 1
                        print("skipped test: " + str(i) + "; Line: " + str(words))
                    except IndexError:
                        print(i)
                        print(line)
                        print(words)
                        break
                    i += 1
        text_file.close()
        sum += 100 * float(right_answers) / float(number_of_tests) #get the percentage
        print_lines.append("Dimension = " + str(dimension) + " Iteration = " + str(iteration) + " Threshold ferq = "+ str(threshold)+" Percentage of correct answers: " + str(sum) + "%")
    
for l in print_lines:
    print(l)