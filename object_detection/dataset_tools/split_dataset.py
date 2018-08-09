import os

image_path = '/home/lz/Lab/TensorFlow/dataset/origin/annotations'
txt_path = '/home/lz/Lab/TensorFlow/dataset/origin/'
file_names = os.listdir(image_path)

num_files = len(file_names)
num_train = int(round(0.6*num_files))
num_test = int(round(0.2*num_files))
print(num_files, num_train, num_test, num_files-num_train-num_test)

tr = open(txt_path + 'train.txt', 'w')
te = open(txt_path + 'test.txt', 'w')
va = open(txt_path + 'val.txt', 'w')
for i in file_names[:num_train]:
    tr.write(i[:-4]+'\n')
for i in file_names[num_train:(num_train+num_test)]:
    te.write(i[:-4]+'\n')
for i in file_names[(num_train + num_test):]:
    va.write(i[:-4]+'\n')

tr.close()
te.close()
va.close()