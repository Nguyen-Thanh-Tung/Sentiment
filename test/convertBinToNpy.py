import numpy as np
from itertools import islice

# arr = np.fromfile('training/wordVectors.bin', dtype=np.float64)
# np.save('training/vectors.npy', arr)
#
# wordsList = np.load('training/vectors.npy')
# print('Loaded the word vector!')
# print(wordsList.shape)

wordsList = np.load('training/wordsList.npy')
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
print(len(wordsList))
# np.save("training/wordsList.txt", wordsList)

# wordsList = [];
# with open("training/wordList.txt", "r") as f:
#     while True:
#         line = list(islice(f, 1))
#         if not line:
#             break
#
#         line = line[0].split()
#         wordsList.append(line[0])
# np.save("training/wordsList", wordsList)

