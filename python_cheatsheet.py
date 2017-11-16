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

# Python generators and the yield keyword 
range
xrange
"""
For the most part, xrange and range are the exact same in terms of functionality. 
They both provide a way to generate a list of integers for you to use, 
however you please. The only difference is that range returns a Python 
list object and xrange returns an xrange object.
What does that mean? Good question! It means that xrange doesn't actually 
generate a static list at run-time like range does. It creates the values 
as you need them with a special technique called yielding. This technique 
is used with a type of object known as generators. If you want to read more 
in depth about generators and the yield keyword, be sure to checkout the 
article Python generators and the yield keyword.
"""
def search(keyword, filename):
    print('generator started')
    f = open(filename, 'r')
    # Looping through the file line by line
    for line in f:
        if keyword in line:
            # If keyword found, return it
            yield line
    f.close()

# Left justify,  Right justify, print command
# %<field width>.<precision>f
for exponent in xrange(7,11):
    print "%-3d%12d" % (exponent,10**exponent)

"""
Encrypts an input string of lowercase letters and prints
the result.  The other input is the distance value.
"""
plainText = raw_input(“Enter a one-word, lowercase message: “)
distance = input(“Enter the distance value: “)
code = “”
for ch in plainText:
    ordValue = ord(ch)
    cipherValue = ordValue + distance
    if cipherValue > ord('z'):
        cipherValue = ord('a') + distance - \
                      (ord('z') - ordValue + 1)
    code +=  chr(cipherValue)
print code

"""
Decrypts an input string of lowercase letters and prints
the result.  The other input is the distance value.
"""
code = raw_input(“Enter the coded text: “)
distance = input(“Enter the distance value: “)
plainText = ''
for ch in code:
    ordValue = ord(ch)
    cipherValue = ordValue - distance
    if cipherValue < ord('a'):
        cipherValue = ord('z') - \
                      (distance - (ord('a') - ordValue + 1))
    plainText += chr(cipherValue)
print plainText

"""
Converts a decimal integer to a string of bits.
"""
decimal = input(“Enter a decimal integer: “)
if decimal == 0: 
    print 0
else:
    print “Quotient Remainder Binary”
    bstring = “”
    while decimal > 0:
        remainder = decimal % 2
        decimal = decimal / 2
        bstring = str(remainder) + bstring
        print “%5d%8d%12s” % (decimal, remainder, bstring)
    print “The binary representation is”, bstring

"""
Converts a string of bits to a decimal integer.
"""
bstring = raw_input(“Enter a string of bits: “)
decimal = 0
exponent = len(bstring) - 1
for digit in bstring:
    decimal = decimal + int(digit) * 2 ** exponent
    exponent = exponent - 1
print “The integer value is”, decimal


#STRING METHOD WHAT IT DOES
s.center(width) #Returns a copy of s centered within the given number of columns.

s.count(sub [, start [, end]]) #Returns the number of non-overlapping occurrences of substring sub in s. Optional
#arguments start and end are interpreted as in slice notation.

s.endswith(sub) #Returns True if s ends with sub or False otherwise.

s.find(sub [, start [, end]]) #Returns the lowest index in s where substring sub is found. Optional arguments
#start and end are interpreted as in slice notation.

s.isalpha() #Returns True if s contains only letters or False otherwise.

s.isdigit() #Returns True if s contains only digits or False otherwise.

s.join(sequence) #Returns a string that is the concatenation of
#the strings in the sequence. The separator between elements is s.

s.lower() #Returns a copy of s converted to lowercase.

s.replace(old, new [, count]) #Returns a copy of s with all occurrences
#of substring old replaced by new. If the optional argument count is given, only the
#first count occurrences are replaced.

s.split([sep]) #Returns a list of the words in s, using sep as the delimiter string. If sep is not specified,
#any whitespace string is a separator.

s.startswith(sub) #Returns True if s starts with sub or False otherwise.

s.strip([aString]) #Returns a copy of s with leading and trailing
#whitespace (tabs, spaces, newlines) removed.If aString is given, remove characters in
#aString instead.

s.upper() #Returns a copy of s converted to uppercase.


# Files
import os
currentDirectoryPath = os.getcwd()
listOfFileNames = os.listdir(currentDirectoryPath)
for name in listOfFileNames:
    if “.py” in name:
        print name


# os MODULE FUNCTION WHAT IT DOES
chdir(path) #Changes the current working directory to path.
getcwd() #Returns the path of the current working directory.
listdir(path) #Returns a list of the names in directory named path.
mkdir(path) #Creates a new directory named path and places it in the current working directory.
remove(path) #Removes the file named path from the current working directory.
rename(old, new) #Renames the file or directory named old to new.
rmdir(path) #Removes the directory named path from the current working directory

# os.path MODULE FUNCTION WHAT IT DOES
exists(path) #Returns True if path exists and False otherwise.
isdir(path) #Returns True if path names a directory and False otherwise.
isfile(path) #Returns True if path names a file and False otherwise.
getsize(path) #Returns the size of the object names by path in bytes.


#LIST METHOD WHAT IT DOES
L.append(element) #Adds element to the end of L.
L.extend(aList) #Adds the elements of aList to the end of L.
L.insert(index, element) #Inserts element at index if index is less than
#the length of L. Otherwise, inserts element at the end of L.
L.pop() #Removes and returns the element at the end of L.
L.pop(index) #Removes and returns the element at index. 
L.sort()

'''
If the data are immutable strings, aliasing can save on memory. But as you
might imagine, aliasing is not always a good thing when side effects are possible.
Assignment creates an alias to the same object rather than a reference to a copy
of the object. To prevent aliasing, a new object can be created and the contents of
the original can be copied to it, as shown in the next session:
'''
third = []
for element in first:
    third.append(element)

#A simpler way to copy a list is to use a slice over all of the positions, as follows:
third = first[:]

# Difference between 'is' and '=='
'''
Python’s is operator can be used to test for object identity. It returns True if the
two operands refer to the exact same object, and it returns False if the operands refer
to distinct objects (even if they are structurally equivalent).
'''
>>> first = [20, 30, 40]
>>> second = first
>>> third = [20, 30, 40]
>>> first == second
True
>>> first == third
True
>>> first is second
True
>>> first is third
False
>>> 

'''
File: median.py
Prints the median of a set of numbers in a file.
'''
fileName = raw_input(“Enter the filename: “)
f = open(fileName, 'r')
    
# Input the text, convert it to numbers, and
# add the numbers to a list
numbers = []
for line in f:
    words = line.split()
    for word in words:
        numbers.append(float(word))
# Sort the list and print the number at its midpoint
numbers.sort()
midpoint = len(numbers) / 2
print “The median is”,
if len(numbers) % 2 == 1:
    print numbers[midpoint]
else:
    print (numbers[midpoint] + numbers[midpoint - 1]) / 2


# Dictionary
# To delete an entry from a dictionary, one removes its key using the method pop.
# Traverse
for key in info:
   print key, info[key] 

#DICTIONARY OPERATION WHAT IT DOES
len(d) #Returns the number of entries in d. aDict[key] Used for inserting a new key, replacing a value, or obtaining a value at an existing key.
d.get(key [, default]) #Returns the value if the key exists or returns the default if the key does not exist. Raises an error if the default is omitted and the key does not exist.
d.pop(key [, default]) #Removes the key and returns the value if the key exists or returns the default if the key does not exist. Raises an error if the default is omitted and the key does not exist.
d.keys() #Returns a list of the keys.
d.values() #Returns a list of the values.

d.items() #Returns a list of tuples containing the keys and values for each entry.
d.has_key(key) #Returns True if the key exists or False otherwise.
d.clear() #Removes all the keys.
for key in d: #key is bound to each key in d in an unspecified order.

# Filtering ?

# Reducing ?

# Jump table ?

# Lambdas ?

# Class and functions
__init__
__eq__
__str__
__cmp__

# Threading, Multi threading
??

# Sort 2D array using 2nd column
movie_year = sorted(movie_year, key=lambda row: row[1], reverse=True)