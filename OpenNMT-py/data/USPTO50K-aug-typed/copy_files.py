
times = 10

with open('src-train-aug-err-old.txt') as f:
	srcs = f.readlines()

with open('tgt-train-aug-err-old.txt') as f:
	tgts = f.readlines()


with open('src-train-aug-err.txt', 'w') as f:
	for t in range(times):
		f.writelines(srcs)

with open('tgt-train-aug-err.txt', 'w') as f:
	for t in range(times):
		f.writelines(tgts)
