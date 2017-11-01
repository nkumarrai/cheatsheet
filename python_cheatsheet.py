with open('layout_list_input.txt' , 'w') as finput:
	with open('layout_list_element_count.txt', 'w') as felement:
		for i,c in zip(input_list, count_list):
			finput.write(i + '\n')
			felement.write(c+ '\n')


import random
training_ids = random.sample(range(0, length_all), training_size)
training_list = [input_list[i] for i in training_ids]
test_list = [input_list[i] for i in range(length_all) if i not in training_ids]
print len(training_list), len(test_list), len(training_list) + len(test_list)

# get all the items from a list using a list of indexes
from operator import itemgetter 
training_list = (itemgetter(*training_ids)(input_list))
print len(training_list), type(list(training_list))
# get rest of the items 
# Set operations
input_list = set(input_list)
test_list = input_list - set(training_list)
test_list = list(test_list)
print len(test_list), type(test_list)


# Iterator - batch processing
def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        batch_size = 500
        for batch_idx in range(0, num_train, batch_size):
            files_batch = train_filepaths[batch_idx:batch_idx+batch_size]
            yield files_batch

# Matching partial list
if any(i in files_batch_val for i in yileded_list):

# Matching subset
set(a) < set(b)