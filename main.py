from fastai.vision.all import *
import pdb
import os

pdb.set_trace()
path = untar_data(URLs.PASCAL_2007)/'images'

df = pd.read_csv(path/'train.csv')
df.head()
