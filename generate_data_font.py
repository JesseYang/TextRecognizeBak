from PIL import Image,ImageDraw,ImageFont
import numpy as np
from scipy import misc
import os
import cv2
import random
import pdb
import shutil
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

raw_data_dir = 'raw_text'
font_dir = 'fonts/fonts_new'
bg_dir = 'background'
generate_count = 200000

min_word_num = 1
max_word_num = 10
max_char_num = 100

min_height = 10
max_height = 30

target_dir = 'generated/test_data'

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


article_paths = [os.path.join(raw_data_dir, e) for e in os.listdir(raw_data_dir) if e.endswith('txt') or e.endswith('TXT')]

font_paths = []
for dirpath, dirnames, filenames in os.walk(font_dir):
    for filename in filenames:
        if filename.endswith('ttf') or filename.endswith('TTF'):
            font_paths.append(os.path.join(dirpath,filename))

random.shuffle(font_paths)

bg_paths = [os.path.join(bg_dir, e) for e in os.listdir(bg_dir)]
print(article_paths[2])
cnt = 0
for idx, font_path in enumerate(font_paths):
    # if idx >= 1:
    #     break
    # font_path = '/home/user/VideoText/recognize_sequences/fonts/fonts_new/Din/DIN_Regular.ttf'
    font_name = font_path.split('/')[-1]
    # if idx >= 2:
    #     break
    print("{}/{}".format(idx+1, len(font_paths)))
    with open(article_paths[2], errors='ignore') as article:
        count = 0
        for line_idx, line in enumerate(article):
            # print("line_idx", line_idx)
            if len(line) == 0:
                continue
            res =  line.split()
            # print(res)
            # pdb.set_trace()

            word_num = random.randint(min_word_num, max_word_num)
            # sublines = chunks(res, word_num)
            subline = res[0:]
            # print(subline)
            text = ' '.join(subline).strip()
            if len(text) > max_char_num:
                text = text[:max_char_num]
            text = ''.join(i for i in text if i in cfg.dictionary).strip()
            if text == '':
                continue


            if random.uniform(0, 1) >= 0.45:
                text = text.upper()
            elif random.uniform(0, 1) >= 0.25:
                text = text.title()
            else:
                text = text.capitalize()
            # print(text)
            # randomly pick a font and height
            # ttf = random.choice(font_paths)
            ttf = font_path
            fg, bg = random.sample(bg_paths, 2)
            h = random.randint(min_height, max_height)
            ttfont = ImageFont.truetype(ttf, h)
            line_w, line_h = ttfont.getsize(text)
            print(ttf)
            # random x,y of text_area
            x = random.randint(0, 3)
            y = random.randint(0, 3)
            # random pad
            h_pad = random.randint(0, 3)
            w_pad = random.randint(0, 3)

            fg = misc.imread(fg, mode = 'L')
            fg = fg // 5
            bg = misc.imread(bg, mode = 'L')
            bg_h, bg_w = bg.shape
            canvas_h, canvas_w = line_h + y + h_pad, line_w + x + w_pad
            if canvas_w <= 0 or canvas_h <= 0:
                continue
            if bg_h > canvas_h and bg_w > canvas_w:
                r_x, r_y = random.randint(0, bg_w - canvas_w - 1), random.randint(0, bg_h - canvas_h - 1)
                canvas = bg[r_y:r_y+canvas_h, r_x:r_x+canvas_w]
            else:
                canvas = cv2.resize(bg, (canvas_w, canvas_h))

            if count == 1:
                break
            count += 1
            fg = cv2.resize(fg, (canvas_w, canvas_h))
            blank = Image.fromarray(np.zeros((canvas_h, canvas_w)) + 255)
            d = ImageDraw.Draw(blank)
            d.text((x, y), text, fill = 0, font = ttfont)
            boolean_mask = np.array(blank) == 0
            canvas[boolean_mask] = fg[boolean_mask]
            tem = font_name+"_"+str(cnt)
            with open('%s/%s.txt' % (target_dir, tem), 'w') as f:
                # f.write(text)
                img = np.asarray(canvas)
                # misc.imsave('%s/%s.png' % (target_dir, tem), img)
            cnt += 1
            # print(cnt)
            if cnt % 1000 == 0:
                print(cnt)
                # quit()
            if cnt >= generate_count:
                quit()
