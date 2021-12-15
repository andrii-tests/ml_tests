import os

dir = '/home/lv-user187/Desktop/heads/workspace/training_demo/images/train/'
names = os.listdir(dir)

for name in names:
	old = os.path.join(dir, name)
	new = os.path.join(dir, name[6:])
	os.rename(old, new)

print('COMPLETED')
