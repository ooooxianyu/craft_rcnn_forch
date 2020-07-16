import os


root = r'D:/AIstudyCode/data/OCR_data/line-04/'

sub_dir = 'line-04'

train_txt = open("data/test.txt",'a',encoding='UTF-8')

for tag in os.listdir(f"{root}/{sub_dir}"):
    file_dir = f"{root}/{sub_dir}/{tag}"
    for file_Name in os.listdir(file_dir):
        file_path = f"{file_dir}/{file_Name}"
        #print(file_Name)
        suffix = file_path.split('.')
        #print(suffix)
        if suffix[1] == 'txt':

            fp = open(file_path,'r',encoding='utf-8')

            img_path = suffix[0]+'.jpg'
            print(img_path)
            train_txt.write(f'{img_path} {fp.readline()}\n')

