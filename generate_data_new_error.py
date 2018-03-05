from PIL import Image,ImageDraw,ImageFont
import numpy as np
from scipy import misc
import os
import cv2
import random
import shutil

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

raw_data_dir = 'raw_text'
font_dir = 'fonts/fonts_new'
bg_dir = 'background'
generate_count = 2000000

min_word_num = 1
max_word_num = 10
max_char_num = 100

min_height = 10
max_height = 30

target_dir = 'generated/test_data_news_error'
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


article_paths = [os.path.join(raw_data_dir, e) for e in os.listdir(raw_data_dir) if e.endswith('txt') or e.endswith('TXT')]
# article_paths.reverse()
random.shuffle(article_paths)
# font_paths = []
# for dirpath, dirnames, filenames in os.walk(font_dir):
#     for filename in filenames:
#         if filename.endswith('ttf') or filename.endswith('TTF'):
#             font_paths.append(os.path.join(dirpath,filename))

# random.shuffle(font_paths)

font_paths = os.listdir(font_dir)
# random.shuffle(font_paths)
print("fonts num: ", len(font_paths))
print(font_paths==['Consola', 'Optima', 'arials', 'Caslons', 'futura', 'AdobeGaramond', 'Bodonis', 'new_roman', 'helvetica', 'adele', 'BookmanOldStyle', 'courier', 'din', 'verdana', 'frutiger'])
# print('Consola', 'Optima', 'arials', 'Caslons', 'futura', 'AdobeGaramond', 'Bodonis', 'new_roman', 'helvetica', 'adele', 'BookmanOldStyle', 'courier', 'din', 'verdana', 'frutiger')
#['Consola', 'Optima', 'arials', 'Caslons', 'futura', 'AdobeGaramond', 'Bodonis', 'new_roman', 'helvetica', 'adele', 'BookmanOldStyle', 'courier', 'din', 'verdana', 'frutiger']
#[0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05]

bg_paths = [os.path.join(bg_dir, e) for e in os.listdir(bg_dir)]

cnt = 0
for idx, article_path in enumerate(article_paths):
    # if idx >10:
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
                font_pro = np.random.choice(font_paths, 1, p=[0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05])[0]
                sub_font_pro = os.path.join(font_dir, font_pro)
                sub_font_pros = os.listdir(sub_font_pro)
                sub_font_pros = [e for e in sub_font_pros if (e.endswith('.ttf') or e.endswith('.TTF')) ]
                # print(sub_font_pros)
                ttf = os.path.join(sub_font_pro, random.choice(sub_font_pros))

                if random.uniform(0, 1) <= 0.4:
                    sub_font_name = ttf.split('/')[-1]#font name
                    ttf = ttf.replace(sub_font_name, font_pro + '_regular.ttf')
                    if not os.path.exists(ttf):
                        ttf = ttf.replace('_regular.ttf', '_regular.TTF')
                # print(ttf)
                if not os.path.exists(ttf):
                    continue
                fg, bg = random.sample(bg_paths, 2)
                h = random.randint(min_height, max_height)
                ttfont = ImageFont.truetype(ttf, h)
                line_w, line_h = ttfont.getsize(text)
               
               	if line_w <=0 or line_h <= 0:
               		continue

                # random x,y of text_area
                x = random.randint(0, 6)
                y = random.randint(0, 6)
                # random pad
                h_pad = random.randint(0, 6)
                w_pad = random.randint(0, 6)
                # print(fg)
                fg = misc.imread(fg, mode = 'L')
                # fg = fg * random.uniform(0.7, 1)
                # misc.imsave("1.jpg", fg)
                if random.uniform(0, 1) >= 0.5:
                    fg = fg // random.randint(2,8)
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
               
                if random.uniform(0, 1) >= 0.5:
                    canvas = canvas + random.uniform(1, 128)

                fg = cv2.resize(fg, (canvas_w, canvas_h))

                if x >= canvas_w or y >= canvas_h:
                    continue
                print("=======")
                print('line_w', line_w, "line_h", line_h)
                print("w_pad", w_pad, "h_pad", h_pad)
                print(x, y, canvas_w, canvas_h)
                print(ttf)
                # print(x, y, canvas_w, canvas_h)
                print(article_path)
               	
               	print(text)

                blank = Image.fromarray(np.zeros((canvas_h, canvas_w)) + 255)
                d = ImageDraw.Draw(blank)
                d.text((x, y), text, fill = 0, font = ttfont)
                boolean_mask = np.array(blank) == 0
                canvas[boolean_mask] = fg[boolean_mask]
                with open('%s/%d.txt' % (target_dir, cnt), 'w') as f:
                    # f.write(text)
                    img = np.asarray(canvas)
                    if random.uniform(0, 1) >= 0.6:
                        img = cv2.GaussianBlur(img,(3,3),1)
                    # misc.imsave('%s/%d.png' % (target_dir, cnt), img)
                cnt += 1
                
                if cnt % 1000 == 0:
                    print(cnt)
                    # quit()
                if cnt >= generate_count:
                    quit()
