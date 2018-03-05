from scipy import misc
import random
import cv2
import numpy as np
import math
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageEnhance

class Augmentor():
    def __init__(self):
        self.rotate_range = 20
        self.min_vignetting = 10
        self.max_vignetting = 200
        self.blur_range = 20
        self.crop_w = 400
        self.crop_h = 400
        self.gaussian_w_kernel_size = 200
        self.gaussian_h_kernel_size = 100
        self.content = ['And', 'grant', 'it', 'Heaven', 'that all', 'who read', 'May', 'find', 'as dear', 'urse at ', 'need And every child', 'who lists my rhyme', \
        'In the bright', 'fireside, nursery clime', 'mayy', 'hear it in as kind a voice', 'As made my childish days rejoice!']
        self.color = ['red', 'blue', 'green', 'black']
    def do(self, img, mask, flage=True):
        if not flage:
            return self.resize_director(img, mask)

        if random.uniform(0, 1) >= 0.5:
            if random.uniform(0, 1) < 0.3:
                img = self.vignetting(img)

            img = self.gaussian_blur(img)
            h, w = img.shape[:2]
            if random.uniform(0, 1) >= 0.8:
                # random center point and focal
                img, mask = self.project_onto_cylinder(img, mask, (w,h//2), 1250)

            img, mask = self.random_resize(img, mask)
            img, mask = self.rotate_and_crop(img, mask)

        img, mask = self.random_crop(img, mask)

        return img, mask

    def project_onto_cylinder(self, img, mask, center, focal):
        """
            Performs a cylindrical projection of a planar image.
        """

        if not focal:
            focal = 750

        # define mapping functions
        scale = focal
        mapX = lambda y, x: focal * np.tan(x/scale)
        mapY = lambda y, x: focal / np.cos(x/scale) * y/scale
        def makeMap(y, x):
            map_x = mapX(y - center[1], x - center[0]) + center[0]
            map_y = mapY(y - center[1], x - center[0]) + center[1]
            return np.dstack((map_x, map_y)).astype(np.int16)
        
        # create the LUTs for x and y coordinates
        map_xy = np.fromfunction(makeMap, img.shape[:2], dtype=np.int16)
        img_mapped = cv2.remap(img, map_xy, None, cv2.INTER_NEAREST)
        mask_mapped = cv2.remap(mask, map_xy, None, cv2.INTER_NEAREST)

        return img_mapped, mask_mapped

    def perspective_transform(self):
        pass
    

    def rotate_and_crop(self, img, mask):           
        """Randomly rotate image and crop the largest rectangle inside the rotated image.

        # Arguments
            imgs: list of processing images.
        """
        def largest_rotated_rect(w, h, angle):
            """
            Get largest rectangle after rotation.
            http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
            """
            angle = angle / 180.0 * math.pi
            if w <= 0 or h <= 0:
                return 0, 0

            width_is_longer = w >= h
            side_long, side_short = (w, h) if width_is_longer else (h, w)

            # since the solutions for angle, -angle and 180-angle are all the same,
            # if suffices to look at the first quadrant and the absolute values of sin,cos:
            sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
            if side_short <= 2. * sin_a * cos_a * side_long:
                # half constrained case: two crop corners touch the longer side,
                #   the other two corners are on the mid-line parallel to the longer line
                x = 0.5 * side_short
                wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
            else:
                # fully constrained case: crop touches all 4 sides
                cos_2a = cos_a * cos_a - sin_a * sin_a
                wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

            return int(np.round(wr)), int(np.round(hr))


        
        deg = random.randrange(-self.rotate_range, self.rotate_range)
        
        res = []
        imgs = [img, mask]
        for img in imgs:
            center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
            rot_m = cv2.getRotationMatrix2D((center[0] - 0.5, center[1] - 0.5), deg, 1)
            ret = cv2.warpAffine(img, rot_m, img.shape[1::-1])
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
            neww, newh = largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
            neww = min(neww, ret.shape[1])
            newh = min(newh, ret.shape[0])
            newx = int(center[0] - neww * 0.5)
            newy = int(center[1] - newh * 0.5)
            res.append(ret[newy:newy + newh, newx:newx + neww])
        
        return res

    def random_crop(self, img, mask):
        h, w = img.shape[:2]
        if h <= self.crop_h or w <= self.crop_w:
            res_img = cv2.resize(img, (self.crop_w, self.crop_h))
            res_mask = cv2.resize(mask, (self.crop_w, self.crop_h))
            return res_img, res_mask
        else:
            y = random.randint(0,h - self.crop_h - 1)
            x = random.randint(0,w - self.crop_w - 1)
            res_img = img[y:y + self.crop_h, x:x + self.crop_w]
            res_mask = mask[y:y + self.crop_h, x:x + self.crop_w]
        
        return res_img, res_mask

    def random_resize(self, img, mask):
        fx = random.uniform(0.5, 2)
        fy = random.uniform(0.5, 2)
        img = cv2.resize(img, None, fx = fx, fy = fy)
        mask = cv2.resize(mask, None, fx = fx, fy = fy)

        return img, mask

    def resize_director(self, img, mask):
        img = cv2.resize(img, (self.crop_w, self.crop_h))
        mask = cv2.resize(mask, (self.crop_w, self.crop_h))

        return img, mask

    def vignetting_rgb(self, imgs):
        print(imgs.shape)
        hs, ws, _ = imgs.shape
        h_gk_size = random.choice(list(range(self.min_vignetting, self.max_vignetting, 2)))
        w_gk_size = random.choice(list(range(self.min_vignetting, self.max_vignetting, 2)))
        h_gk = cv2.getGaussianKernel(hs,h_gk_size)
        w_gk = cv2.getGaussianKernel(ws,w_gk_size)
        c = h_gk*w_gk.T
            # e = img*channelc
        d = c/c.max()
        # channel = np.ones((hs,ws,3), dtype=np.int)
        # channel[:,:,0] = imgs[:,:,0]*d
        # channel[:,:,1] = imgs[:,:,1]*d
        # channel[:,:,2] = imgs[:,:,2]*d
        imgs[:,:,0] = imgs[:,:,0]*d
        imgs[:,:,1] = imgs[:,:,1]*d
        imgs[:,:,2] = imgs[:,:,2]*d

        return imgs

    def add_text_line_rgb(self, imgs_path, is_rotate=False, is_line=False):
        img = Image.open(imgs_path)
        h, w, _ = np.asarray(img).shape
        draw = ImageDraw.Draw(img)
        newfont=ImageFont.truetype('font.TTF',random.randint(40,500))
        coor = random.randint(0, int(w*0.9)), random.randint(0, int(h*0.9))
        # print(coor)
        draw.text(coor, random.choice(self.content), fill=random.choice(self.color), font = newfont)
        # h, w, _ = img.shape
        # print(img.shape)
        if is_line:
            x1, y1 = random.randint(0, int(w*0.5)), random.randint(0, int(h*0.9)) 
            x2, y2 = random.randint(int(w*0.5), int(w*0.8)), random.randint(int(h*0.5), h)
            x3, y3 = random.randint(int(w*0.6), int(w*0.9)), random.randint(int(h*0.6), h)
            draw.line(((x1, y1),(x2, y2),(x3, y3)), fill=random.choice(self.color), width=random.randint(2,15))
        if is_rotate:
            img = img.rotate(random.choice([45, -45]))
        # img = img.filter(ImageFilter.BLUR)

        # misc.imsave('tem_dir/img_blur.png', np.array(img))
        return np.asarray(img)

    def vignetting(self, img):
        h, w = img.shape
        h_gk_size = random.choice(list(range(self.min_vignetting,self.max_vignetting, 2)))
        w_gk_size = random.choice(list(range(self.min_vignetting,self.max_vignetting, 2)))
        h_gk = cv2.getGaussianKernel(h,h_gk_size)
        w_gk = cv2.getGaussianKernel(w,w_gk_size)
        c = h_gk*w_gk.T
        d = c/c.max()
        e = img*d
        
        return e

    def brightness(self, imgs_path, is_enhance=False, is_contrast=False):
        image = Image.open(imgs_path)

        if is_enhance and random.uniform(0, 1) > 0.5:
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice([1.5, 0.5])
            image = enh_bri.enhance(brightness) 

        if is_contrast and random.uniform(0, 1) > 0.6:
            enh_con = ImageEnhance.Contrast(image)  
            contrast = 0.5  
            image = enh_con.enhance(contrast) 
        return np.asarray(image)

    def gaussian_blur(self, img):
        h_size = random.choice(list(range(1, self.gaussian_h_kernel_size, 2)))
        w_size = random.choice(list(range(1, self.gaussian_w_kernel_size, 2)))
        sigma = random.uniform(1.9, 2.5)
        img = cv2.GaussianBlur(img, (h_size, w_size), sigma)
        return img

    
if __name__ == '__main__':
    aug = Augmentor()
    img = misc.imread('dpengcan.jpg', mode = 'RGB')
    # img = misc.imread('dinput_crab4.png', mode = 'RGB')
    # mask = misc.imread('tem_dir/1.jpg', mode = 'L')
    # img = Image.open('tem_dir/1.jpg')
    # img=img.convert("L")
    # img = aug.brightness('tem_dir/1.jpg', True, True)
    # print(img.size,mask.shape)
    # print(np.array(img).shape)
    # img = img.rotate(random.choice([360]))
    # img_blur = aug.gaussian_blur(img)
    img_blur = aug.vignetting_rgb(img)
    # img_blur = aug.add_text_line_rgb('tem_dir/1.jpg')
    misc.imsave('00.png', img_blur)
    # misc.imsave('tem_dir/img_blur11.png', mask)


    # img_res, mask_res = aug.do(img, mask)

    # misc.imsave('test/img_output.png', img_res)
    # misc.imsave('test/mask_output.png', mask_res)
    # 