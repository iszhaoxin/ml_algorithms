import os

def idem_mkdir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

def gen_dir_file(dir_path):
	for d in sorted(os.listdir(source_dir)):
		if '.DS' in d: continue
		for f in sorted(os.listdir(source_dir+'/'+d)):
			if '.DS' in d: continue
			yield d,f

def init_dir(source_dir,target_dir):
	idem_mkdir(target_dir)
	for d,_ in gen_dir_file(source_dir):
		idem_mkdir(target_dir+'/'+d)

def get_as_dict(line):
	d = dict()
	sp = line.strip().split(' ')
	for item in sp[:-1]:
		term, value = item.split(':')
		d[term] = value
	if sp[-1].split(':')[-1] == "negative":
		return d, "neg"
	elif sp[-1].split(':')[-1] == "positive":
		return d, "pos"


source_dir = 'processed_acl'
target_dir = 'data'

init_dir(source_dir,target_dir)
word2id = dict()
id2word = dict()
wordCnt = dict()
cnt = 0
for d,f in gen_dir_file(source_dir):
	converted_datum = list()
	if d == "books":
		for l in open('/'.join([source_dir,d,f])):
			converted_data = list()
			get_dict, label = get_as_dict(l)
			cnt += len(get_dict)
			for k,v in get_dict.items():
				if k not in word2id: word2id[k] = str(len(word2id))
				if word2id[k] not in id2word: id2word[word2id[k]] = k
				converted_data.append(word2id[k]+':'+v)
			converted_data.append(label)
			converted_datum.append(' '.join(converted_data))
		with open('/'.join([target_dir,d,f]),'w') as fp:
			fp.write('\n'.join(converted_datum))
			print(len(word2id))
			print(cnt)
