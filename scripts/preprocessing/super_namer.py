import os
from bs4 import BeautifulSoup

dir = '/home/lv-user187/Desktop/heads/workspace/training_demo/images/train/'
names = os.listdir(dir)

for name in names:
	try:
		if name[-4:] == '.xml':
			full_path = os.path.join(dir, name)
			
			with open(full_path, 'r') as f:
				data = f.read()

			bs_data = BeautifulSoup(data, "xml")
			target_name = bs_data.find('filename').text
			full_target_name = os.path.join(dir, target_name)
		

			old_name = name[:-4] + '.jpg'
			full_old_name = os.path.join(dir, old_name)

			os.rename(full_old_name, full_target_name)
	except:
		pass
print('SUCCESS')
