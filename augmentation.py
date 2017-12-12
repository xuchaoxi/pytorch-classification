import numpy as np
import cv2

def enhance(im): 
    lab= cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,30))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    im_new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_new

class Augmentation(object):
    def resize(self,im,size=(512,512),e=1e-15):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        im_min, im_max = np.min(im),np.max(im)
        im_std = (im - im_min) / (im_max - im_min + e)
        resized_std = cv2.resize(im_std, size)
        resized_im = resized_std * (im_max - im_min) + im_min
        return resized_im

    def rotate(self,im,rotation_param,keep_aspect_ratio):
        h,w,_ = im.shape
        cX,cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX,cY),-rotation_param,1.0)
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])

        if not keep_aspect_ratio:
            M[0,2] += (w // 2) - cX
            M[1,2] += (h // 2) - cY
            im_new = cv2.warpAffine(np.array(im,dtype=np.float32),M,(w,h))       
            return im_new
        
        nW = int(h*sin + w*cos)
        nH = int(h*cos + w*sin)
        M[0,2] += (nW // 2) - cX
        M[1,2] += (nH // 2) - cY
        im_new = cv2.warpAffine(np.array(im,dtype=np.float32),M,(nW,nH))
        
        x0 = int(max(0,(nW-w)/2))
        x1 = int(min((nW+w)/2,nW))
        y0 = int(max(0,(nH-h)/2))
        y1 = int(min((nH+h)/2,nH))
        return im_new[y0:y1, x0:x1]
        
        
    def flip(self,im):
        return cv2.flip(im,1)

    def zoom(self,im,w_dev):
        h,w,_ = im.shape
        #TODO
        #h_dev = int(..) keep aspect ratio
        return im[w_dev:h-w_dev, w_dev:w-w_dev]

    def crop(self,im,w0=0,w1=0,h0=0,h1=0):
        h,w,_ = im.shape
        return im[h0:h-h1, w0:w-w1]

    def enhance(self,im): 
        lab= cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,30))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        im_new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return im_new

    def clip(self,im):
        im[im>255] = 255
        im[im<0] = 0
        return im

    def brightness(self,im,alpha):
        im *= alpha
        return im

    def contrast(self,im,alpha):
        coef = np.array([[[0.299, 0.587,0.114]]])
        gray = im * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        im *= alpha
        im += gray
        return im
   
    def saturation(self,im,alpha):
        coef = np.array([[[0.299, 0.587,0.114]]])
        gray = im * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        im *= alpha
        im += gray
        return im

    def gamma_trans(self,im,alpha):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        im /= 255
        f = lambda x:x**alpha
        im_trans = f(im)
        return im_trans*255

    def multiple_rgb(self,im,alphas):
        for i in xrange(len(alphas)):
            im[:,:,[i]] *= alphas[i]
        return im

class OurAug(Augmentation):
   
    def __init__(self, params):
        self.aug_cfg = params

    def process(self, img, rand_values=None):
        chosen_value = {}

        im = np.copy(img)
        output_shape = (self.aug_cfg['out_h'],self.aug_cfg['out_w'])
        if  self.aug_cfg.get('rotation',False):
            if rand_values:
                rotate_params = rand_values['rotate_params']
            else:
                rotate_params = np.random.randint(self.aug_cfg['rotation_range'][0],
                                    self.aug_cfg['rotation_range'][1])
                chosen_value['rotate_params'] = rotate_params
            im = self.rotate(im,rotate_params,self.aug_cfg["keep_aspect_ratio"])


        if self.aug_cfg.get('crop',False):
            if rand_values:
                do_crop = rand_values['do_crop']
            else:
                do_crop = self.aug_cfg['crop_prob'] > np.random.rand()
                chosen_value['do_crop'] = do_crop

            if do_crop:

                if rand_values:
                    w0, w1 = rand_values['w0'], rand_values['w1']
                    h0, h1 = rand_values['h0'], rand_values['h1']
                else:
                    h,w,_ = im.shape
                    w_dev = int(self.aug_cfg['crop_w'] * w)
                    h_dev = int(self.aug_cfg['crop_h'] * h)
                
                    w0 = np.random.randint(0, w_dev + 1)
                    w1 = np.random.randint(0, w_dev + 1)
                    h0 = np.random.randint(0, h_dev + 1)
                    h1 = np.random.randint(0, h_dev + 1)

                    chosen_value['w0'] = w0
                    chosen_value['w1'] = w1
                    chosen_value['h0'] = h0
                    chosen_value['h1'] = h1
                im = self.crop(im,w0,w1,h0,h1)

        if self.aug_cfg.get('gamma',False):
            if rand_values:
                pass
            else:
                gamma_options = self.aug_cfg['gamma_options'] 
                rand = np.random.randint(0,len(gamma_options))
                gamma_param = gamma_options[rand]
                chosen_value['gamma_param'] = gamma_param
                im = self.gamma_trans(im,gamma_param)

        if self.aug_cfg.get('contrast',False):
            if rand_values:
                pass
            else:
                contrast_param = np.random.uniform(self.aug_cfg['contrast_range'][0],
                                self.aug_cfg['contrast_range'][1])
                chosen_value['contrast_param'] = contrast_param
                im = self.contrast(im,contrast_param)            


        if self.aug_cfg.get('brightness',False):
            if rand_values:
                pass
            else:

                brightness_param = np.random.uniform(self.aug_cfg['brightness_range'][0],
                                self.aug_cfg['brightness_range'][1])
                chosen_value['brightness_param'] = brightness_param
                im = self.brightness(im,brightness_param)            

        if self.aug_cfg.get('saturation',False):
            if rand_values:
                pass
            else:
                saturation_param = np.random.uniform(self.aug_cfg['saturation_range'][0],
                                self.aug_cfg['saturation_range'][1])
                chosen_value['saturation_param'] = saturation_param
                im = self.saturation(im, saturation_param)            

        if self.aug_cfg.get('flip',False):
            if rand_values:
                do_flip = rand_values['do_flip']
            else:
                do_flip = self.aug_cfg['flip_prob'] > np.random.rand()
                chosen_value['do_flip'] = do_flip
           
            if do_flip:
                im = self.flip(im)
               
        if self.aug_cfg.get('zoom',False):
            if rand_values:
                do_zoom = rand_values['do_zoom']
            else:
                do_zoom = self.aug_cfg['zoom_prob'] > np.random.rand()
                chosen_value['do_zoom'] = do_zoom
           
            if do_zoom:
                if rand_values:
                    w_dev = rand_values['w_dev']
                else:
                    zoom_min, zoom_max = self.aug_cfg['zoom_range']
                    h,w,_ = im.shape
                    w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * w)
                    chosen_value['w_dev'] = w_dev
                im = self.zoom(im,w_dev)

        if tuple(im.shape[:2]) != output_shape:
            im = self.resize(im,output_shape)
        return im, chosen_value