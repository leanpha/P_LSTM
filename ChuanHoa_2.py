# -*- coding: utf-8 -*-
import re
import datetime
import threading
try:
	from Queue import Queue
except Exception as e:
	from queue import Queue

Num_Thread = 50
'''Dieu chi so thread cho phu hop.
Neu file ngan ma so thread qua lon cung se bi cham.
Do qua ton thoi gian khoi tao thread'''
print("Waiting.......")
s = datetime.datetime.now()
#path = "/home/hoailam/Videos/KhoaLuan/"
path = "D:/dclvtn_research/w2v/"
filein = path + "input.txt"
fileout = path + "out1.txt"

pt = "[^AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬBCDĐEÈẺẼÉẸÊỀỂỄẾỆFGHIÌỈĨÍỊKLMNOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢPQRSTUÙỦŨÚỤƯỪỬỮỨỰVWXYỲỶỸÝỴZaàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789/.?,:;!\n\s]";
f1 = open(fileout,'w')
#f = open(path + 'vne.txt', 'r')
exitFlag = 0
def replace_char(line):
	#Ky tu dac biet dung ky tu thay the
	line = line.replace("&nbsp;"," "); # cach
	line = line.replace("&gt;","");   # >
	line = line.replace("&lt;","");   # <
	line = line.replace("&amp;","");  # &
	line = line.replace("&cent;",""); # ¢
	line = line.replace("&pound;","");# £
	line = line.replace("&yen;","");  # ¥
	line = line.replace("&euro;",""); # €
	line = line.replace("&copy;",""); # ©
	line = line.replace("&reg;","");  # ®
	line = line.replace("&rsquo;","");  # '
	line = line.replace("&ldquo;","");  # "
	line = line.replace("&lsquo;","");  # '
	line = line.replace("&rdquo;","");  # "
	line = line.replace("&ndash;"," "); # -
	#Chu dung ky tu thay the
	line = line.replace("&Agrave;","À");
	line = line.replace("&agrave;","à");
	line = line.replace("&Egrave;","È");
	line = line.replace("&egrave;","è");
	line = line.replace("&Eacute;","É");
	line = line.replace("&eacute;","é");
	line = line.replace("&Iacute;","Í");
	line = line.replace("&iacute;","í");
	line = line.replace("&Igrave;","Ì");
	line = line.replace("&igrave;","ì");
	line = line.replace("&Acirc;","Â");
	line = line.replace("&acirc;","â");
	line = line.replace("&#258;","Ă");
	line = line.replace("&#259;","ă");
	line = line.replace("&Atilde;","Ã");
	line = line.replace("&atilde;","ã");
	line = line.replace("&Otilde;","Õ");
	line = line.replace("&otilde;","õ");
	line = line.replace("&Ugrave;","Ù");
	line = line.replace("&ugrave;","ù");
	line = line.replace("&Uacute;","Ú");
	line = line.replace("&uacute;","ú");
	line = line.replace("&Ecirc;","Ê");
	line = line.replace("&ecirc;","ê");
	line = line.replace("&#272;","Đ");
	line = line.replace("&#273;","đ");
	line = line.replace("&Ograve;","Ò");
	line = line.replace("&ograve;","ò");
	line = line.replace("&Oacute;","Ó");
	line = line.replace("&oacute;","ó");
	line = line.replace("&Ocirc;","Ô");
	line = line.replace("&ocirc;","ô");
	line = line.replace("&Yacute;","Ý");
	line = line.replace("&yacute;","ý");
	line = line.replace("&Aacute;","Á");
	line = line.replace("&aacute;","á");
	line = line.replace("&hellip;","...");

	line = line.replace("&#192;","À");
	line = line.replace("&#193;","Á");
	line = line.replace("&#194;","Â");
	line = line.replace("&#226;","â");
	line = line.replace("&#195;","Ã");
	line = line.replace("&#227;","ã");
	line = line.replace("&#200;","È");
	line = line.replace("&#232;","è");
	line = line.replace("&#201;","É");
	line = line.replace("&#233;","v");
	line = line.replace("&#202;","Ê");
	line = line.replace("&#234;","ê");
	line = line.replace("&#204;","Ì");
	line = line.replace("&#236;","ì");
	line = line.replace("&#205;","Í");
	line = line.replace("&#237;","í");
	line = line.replace("&#208;","Ð");
	line = line.replace("&#210;","Ò");
	line = line.replace("&#211;","Ó");
	line = line.replace("&#212;","Ô");
	line = line.replace("&#244;","ô");
	line = line.replace("&#213;","Õ");
	line = line.replace("&#245;","õ");
	line = line.replace("&#217;","Ù");
	line = line.replace("&#218;","Ú");
	line = line.replace("&#221;","Ý");
	line = line.replace("&#224;","à");
	line = line.replace("&#225;","á");
	line = line.replace("&#296;","Ĩ");
	line = line.replace("&#272;","Đ");
	line = line.replace("&#273;","đ");
	line = line.replace("&#242;","ò");
	line = line.replace("&#243;","ó");
	line = line.replace("&#245;","õ");
	line = line.replace("&#249;","ù");
	line = line.replace("&#250;","ú");
	line = line.replace("&#253;","ý");
	line = line.replace("&#296;","Ĩ");
	line = line.replace("&#297;","ĩ");
	line = line.replace("&#305;","ı");
	line = line.replace("&#360;","Ũ");
	line = line.replace("&#297;","ĩ");
	line = line.replace("&#361;","ũ");
	line = line.replace("&#416;","Ơ");
	line = line.replace("&#417;","ơ");
	line = line.replace("&#432;","ư");
	line = line.replace("&#461;","Ǎ");
	line = line.replace("&#462;","ǎ");
	line = line.replace("&#7841;","ạ");
	line = line.replace("&#7842;","Ả");
	line = line.replace("&#7843;","ả");
	line = line.replace("&#7844;","Ấ");
	line = line.replace("&#7845;","ấ");
	line = line.replace("&#7846;","Ầ");
	line = line.replace("&#7847;","ầ");
	line = line.replace("&#7848;","Ẩ");
	line = line.replace("&#7849;","ẩ");
	line = line.replace("&#7850;","Ẫ");
	line = line.replace("&#7852;","Ậ");
	line = line.replace("&#7854;","Ắ");
	line = line.replace("&#7856;","Ằ");
	line = line.replace("&#7858;","Ẳ");
	line = line.replace("&#7860;","Ẵ");
	line = line.replace("&#7861;","ẵ");
	line = line.replace("&#7862;","Ặ");
	line = line.replace("&#7864;","Ẹ");
	line = line.replace("&#7865;","ẹ");
	line = line.replace("&#7866;","Ẻ");
	line = line.replace("&#7868;","Ẽ");
	line = line.replace("&#7869;","ẽ");
	line = line.replace("&#7870;","Ế");
	line = line.replace("&#7851;","ẫ");
	line = line.replace("&#7853;","ậ");
	line = line.replace("&#7855;","ắ");
	line = line.replace("&#7857;","ằ");
	line = line.replace("&#7859;","ẳ");
	line = line.replace("&#7863;","ặ");
	line = line.replace("&#7867;","ẻ");
	line = line.replace("&#7871;","ế");
	line = line.replace("&#7872;","Ề");
	line = line.replace("&#7873;","ề");
	line = line.replace("&#7874;","Ể");
	line = line.replace("&#7875;","ể");
	line = line.replace("&#7876;","Ễ");
	line = line.replace("&#7877;","ễ");
	line = line.replace("&#7878;","Ệ");
	line = line.replace("&#7879;","ệ");
	line = line.replace("&#7880;","Ỉ");
	line = line.replace("&#7881;","ỉ");
	line = line.replace("&#7882;","Ị");
	line = line.replace("&#7883;","ị");
	line = line.replace("&#7884;","Ọ");
	line = line.replace("&#7885;","ọ");
	line = line.replace("&#7886;","Ỏ");
	line = line.replace("&#7887;","ỏ");
	line = line.replace("&#7888;","Ố");
	line = line.replace("&#7889;","ố");
	line = line.replace("&#7890;","Ồ");
	line = line.replace("&#7891;","ồ");
	line = line.replace("&#7892;","Ổ");
	line = line.replace("&#7893;","ổ");
	line = line.replace("&#7894;","Ỗ");
	line = line.replace("&#7895;","ỗ");
	line = line.replace("&#7896;","Ộ");
	line = line.replace("&#7897;","ộ");
	line = line.replace("&#7898;","Ớ");
	line = line.replace("&#7899;","ớ");
	line = line.replace("&#7900;","Ờ");
	line = line.replace("&#7901;","ờ");
	line = line.replace("&#7902;","Ở");
	line = line.replace("&#7903;","ở");
	line = line.replace("&#7904;","Ỡ");
	line = line.replace("&#7905;","ỡ");
	line = line.replace("&#7906;","Ợ");
	line = line.replace("&#7907;","ợ");
	line = line.replace("&#7908;","Ụ");
	line = line.replace("&#7909;","ụ");
	line = line.replace("&#7910;","Ủ");
	line = line.replace("&#7911;","ủ");
	line = line.replace("&#7912;","Ứ");
	line = line.replace("&#7913;","ứ");
	line = line.replace("&#7914;","Ừ");
	line = line.replace("&#7915;","ừ");
	line = line.replace("&#7916;","Ử");
	line = line.replace("&#7917;","ử");
	line = line.replace("&#7918;","Ữ");
	line = line.replace("&#7919;","ữ");
	line = line.replace("&#7920;","Ự");
	line = line.replace("&#7921;","ự");
	line = line.replace("&#7922;","Ỳ");
	line = line.replace("&#7923;","ỳ");
	line = line.replace("&#7924;","Ỵ");
	line = line.replace("&#7925;","ỵ");
	line = line.replace("&#7926;","Ỷ");
	line = line.replace("&#7927;","ỷ");
	line = line.replace("&#7928;","Ỹ");
	line = line.replace("&#7929;","ỹ");

	return line

class myThread (threading.Thread):
	def __init__(self, threadID, name, q):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.q = q
	def run(self):
		print(self.name + " starting...")
		process_data(self.name, self.q)

def process_data(threadName, q):
	global exitFlag
	while exitFlag == 0:
		if (q.empty() == False):
			data = q.get()
			function(data)
			q.task_done()
		else:
			print(threadName + " completed !")


def function(line):
	line = replace_char(line)

	index = line.find('http')
	while index > -1 :
		i = index - 1
		while i >= 0:
			if(line[i] == " "):
				break
			i -= 1
		if i == -1:
			i = 0
		j = index + 1
		while j < len(line):
			if(line[j] == " "):
				break
			j += 1
		if j > len(line):
			j = len(line)
		s = line[i:j]
		line = line.replace(s, " ")
		index = line.find('http')

	index = line.find('www')
	while index > -1 :
		i = index - 1
		while i >= 0:
			if(line[i] == " "):
				break
			i -= 1
		if i == -1:
			i = 0
		j = index + 1
		while j < len(line):
			if(line[j] == " "):
				break
			j += 1
		if j > len(line):
			j = len(line)
		s = line[i:j]
		line = line.replace(s, " ")
		index = line.find('www')
	index = line.find('@')
	while index > -1 :
		i = index - 1
		while i >= 0:
			if(line[i] == " "):
				break
			i -= 1
		if i == -1:
			i = 0
		j = index + 1
		while j < len(line):
			if(line[j] == " "):
				break
			j += 1
		if j > len(line):
			j = len(line)
		s = line[i:j]
		line = line.replace(s, " ")
		index = line.find('@')
	line = re.sub(pt, '', line)
	line = ' '.join(line.split())
	f1.write(line + '\n')
def Main():
	threads = []
	global exitFlag
	exitFlag = 0
	workQueue = Queue(Num_Thread)
	queueLock = threading.Lock()
	# Tao cac thread moi
	for i in range(Num_Thread):
		thread = myThread(i, 'Thread-'+str(i), workQueue)
		thread.daemon = True
		thread.start()
		threads.append(thread)

	# Dien vao queue
	queueLock.acquire()
	with open(filein, 'r') as f:
		for line in f:
			workQueue.put(line)
	queueLock.release()

	# Doi den khi queue la trong
	while not workQueue.empty():
		pass

	# Thong bao cho thread do la thoi gian de ket thuc
	exitFlag = 1

	# Doi cho tat ca thread ket thuc
	for t in threads:
		t.join()
	f.close()
	f1.close()
	e = datetime.datetime.now()
	print("######################################################################")
	print("#{:^68}#".format(" "))
	print("#{:^68}#".format("Start: " + str(s)))
	print("#{:^68}#".format("End  : " + str(e)))
	print("#{:^68}#".format("Total: " + str(e-s)))
	print("#{:^68}#".format(" "))
	print("######################################################################")
	print("Done !")

if __name__ == '__main__':
	Main()