import numpy as np
read_dictionary = np.load('similarity_vocab.npy',allow_pickle=True,encoding="latin1").item()
print(read_dictionary[u'man'][0][0])