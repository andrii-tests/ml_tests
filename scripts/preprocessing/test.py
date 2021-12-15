import os

print('heads: ', len(
    os.listdir('../../datasets/HollywoodHeads/Part_Heads_5')))
print('train: ', len(
    os.listdir('../../datasets/HollywoodHeads/Part_Heads_5/train')))
print('test : ', len(
    os.listdir('../../datasets/HollywoodHeads/Part_Heads_5/test')))
