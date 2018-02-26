file1 = open('breast-cancer-wisconsin.txt', 'r')
file2 = open('breast-cancer-wisconsin-fixed.txt', 'w')
for line in file1:
    if('?' not in line):
        file2.write(line)
file1.close()
file2.close()
