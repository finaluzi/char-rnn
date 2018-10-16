import collections
import codecs
import sys

input_file = sys.argv[1]
frec = int(sys.argv[2])

with codecs.open(input_file, "r", encoding='utf-8') as f:
    data = f.read()
counter = collections.Counter(data)
c2 = counter.most_common()
l2 = []
for z in c2:
    if z[1] >= frec:
        l2.append(z[0])
    else:
        break
print(len(l2))


f_out = codecs.open(input_file + 'c2', "w", encoding='utf-8')
with codecs.open(input_file, "r", encoding='utf-8') as f2:
    for line in f2:
        line_out = ''
        for char in line:
            if char in l2:
                line_out += char
            else:
                line_out += '@'
        f_out.write(line_out)
f_out.close()

# hey = [' '.join([j for j in i if counter[j] > 1]) for i in data]
# with codecs.open(input_file+'c2', "w", encoding='utf-8') as f2:
#     f2.write(hey)
