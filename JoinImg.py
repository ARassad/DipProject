import sys
from PIL import Image
from tqdm import tqdm

for i in tqdm(range(560)):
    tmp = ('000000' + str(i))[-5:] + ".png"
    images = map(Image.open, ['A:\\Diplom\\Project\\datasets\\men\\train_img\\' + tmp,
                              'A:\\Diplom\\Project\\datasets\\men\\train_label\\' + tmp])

    images = [i.resize((256, 256)) for i in images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save('A:\\Diplom\\Project\\datasets\\men\\train\\' + tmp)
