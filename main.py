from fastai.vision.all import *
import pdb
import pathlib
import matplotlib.image as mpimg

path = pathlib.Path(__file__).parent.absolute()
image_files = get_image_files(path/'images')

def get_ctr(f):
    c1 = int(re.findall(r'(.+)x', f.name)[0]) 
    c2 = int(re.findall(r'x(.+)\.', f.name)[0])
    return tensor([c1, c2]) 

biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter = RandomSplitter(seed=42),
    item_tfms=Resize(128),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)

dls = biwi.dataloaders(path/'images', bs = 5)
pdb.set_trace()
print(0)
