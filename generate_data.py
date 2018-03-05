from PIL import Image,ImageDraw,ImageFont
import numpy as np
from scipy import misc
import os
import cv2
import random


try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

raw_data_dir = 'raw_text'
font_dir = 'fonts/fonts_new'
bg_dir = 'background'
generate_count = 1500000

min_word_num = 5
max_word_num = 10
max_char_num = 200
min_char_num = 10

min_height = 10
max_height = 30

target_dir = 'generated/hard_sammple_minning'

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

cnt = 0
for idx, article_path in enumerate(article_paths):
    # if idx > 0:
    #     break
    print("{}/{}".format(idx+1, len(article_paths)))
    with open(article_path, errors='ignore') as article:
        for line in article:
            
            res =  line.split()
            word_num = random.randint(min_word_num, max_word_num)
            sublines = chunks(res, word_num)
            for subline in sublines:
                text = ' '.join(subline).strip()
                if len(text) > max_char_num:
                    text = text[:max_char_num]
                text = ''.join(i for i in text if i in cfg.dictionary).strip()
                if len(text) < min_char_num:
                    continue
                if random.uniform(0, 1) >= 0.45:
                	text = text.upper()
                elif random.uniform(0, 1) >= 0.25:
                	text = text.title()
                else:
                	text = text.capitalize()
                # print(text)
                # randomly pick a font and height
                ttf = random.choice(font_paths)

                if random.uniform(0, 1) <= 0.4:
                	font_name = ttf.split('/')[-2]#font name
                	sub_font_name = ttf.split('/')[-1]#font name
                	ttf = ttf.replace(sub_font_name, font_name + '_regular.ttf')
                	if not os.path.exists(ttf):
                		ttf = ttf.replace('_regular.ttf', '_regular.TTF')

                fg, bg = random.sample(bg_paths, 2)
                h = random.randint(min_height, max_height)
                ttfont = ImageFont.truetype(ttf, h)
                line_w, line_h = ttfont.getsize(text)
                if line_h <= 0 or line_w <= 0:
                    continue
                # random x,y of text_area
                x = random.randint(0, 3)
                y = random.randint(0, 3)
                # random pad
                h_pad = random.randint(1, 3)
                w_pad = random.randint(1, 3)

                fg = misc.imread(fg, mode = 'L')
                fg = fg // 5
                bg = misc.imread(bg, mode = 'L')
                bg_h, bg_w = bg.shape
                canvas_h, canvas_w = line_h + y + h_pad, line_w + x + w_pad

                if bg_h > canvas_h and bg_w > canvas_w:
                    r_x, r_y = random.randint(0, bg_w - canvas_w - 1), random.randint(0, bg_h - canvas_h - 1)
                    canvas = bg[r_y:r_y+canvas_h, r_x:r_x+canvas_w]
                else:
                    canvas = cv2.resize(bg, (canvas_w, canvas_h))
                fg = cv2.resize(fg, (canvas_w, canvas_h))
                blank = Image.fromarray(np.zeros((canvas_h, canvas_w)) + 255)
                d = ImageDraw.Draw(blank)
                d.text((x, y), text, fill = 0, font = ttfont)
                boolean_mask = np.array(blank) == 0
                canvas[boolean_mask] = fg[boolean_mask]
                with open('%s/%d.txt' % (target_dir, cnt), 'w') as f:
                    f.write(text)
                    img = np.asarray(canvas)
                    misc.imsave('%s/%d.png' % (target_dir, cnt), img)
                cnt += 1
               
                if cnt % 1000 == 0:
                    print(cnt)
                    # quit()
                if cnt >= generate_count:
                    quit()
