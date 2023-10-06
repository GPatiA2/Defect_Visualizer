import cv2
import numpy as np

class Image:

    def __init__(self, im, name, shape, bboxes):
        
        self.name = name
        self.orig_image = im
        self.use_image = self.orig_image.copy()
        self.show_name = False
        self.show_bboxes = False
        self.show_temperature = False
        self.show_defect = False
        self.bboxes = bboxes
        self.shape = shape

    def reset_image(self):
        self.use_image = self.orig_image.copy()
        im = self.use_image.copy()
         
        if self.show_bboxes:
            print(len(self.bboxes))
            im = cv2.drawContours(im, self.bboxes, -1, (0, 0, 255), 2)
        
        return im

    def filter_hue(self, h_min, h_max):

        img = self.orig_image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_min = cv2.threshold(img[:,:,0], h_min, 360, cv2.THRESH_BINARY)[1]   
        img_max = cv2.threshold(img[:,:,0], h_max, 360, cv2.THRESH_BINARY_INV)[1]     

        img = np.uint8(np.logical_and(img_min, img_max)) * 255

        img = np.stack((img,)*3, axis=-1)

        if self.show_name:
            img = cv2.putText(img, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.show_bboxes:
            img = cv2.drawContours(img, self.bboxes, -1, (0, 0, 255), 2)
        
        return img

    def filter_sat(self, s_min, s_max):

        img = self.orig_image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_min = cv2.threshold(img[:,:,1], s_min, 255, cv2.THRESH_BINARY)[1]
        img_max = cv2.threshold(img[:,:,1], s_max, 255, cv2.THRESH_BINARY_INV)[1]

        img = np.uint8(np.logical_and(img_min, img_max)) * 255

        img = np.stack((img,)*3, axis=-1)

        if self.show_name:
            img = cv2.putText(img, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.show_bboxes:
            img = cv2.drawContours(img, self.bboxes, -1, (0, 0, 255), 2)

        return img

    def filter_val(self, v_min, v_max):

        img = self.orig_image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_min = cv2.threshold(img[:,:,2], v_min, 255, cv2.THRESH_BINARY)[1]
        img_max = cv2.threshold(img[:,:,2], v_max, 255, cv2.THRESH_BINARY_INV)[1]

        img = np.uint8(np.logical_and(img_min, img_max)) * 255

        img = np.stack((img,)*3, axis=-1)

        if self.show_name:
            img = cv2.putText(img, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.show_bboxes:
            img = cv2.drawContours(img, self.bboxes, -1, (0, 0, 255), 2)

        return img

    def filter_image(self, act_values):

        img = self.orig_image.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img)

        h = self.filter_hue(act_values['hue'][0], act_values['hue'][1])
        s = self.filter_sat(act_values['saturation'][0], act_values['saturation'][1])
        v = self.filter_val(act_values['value'][0], act_values['value'][1])

        img = np.uint8(np.logical_and(h,s,v)) * 255

        img = np.stack((img,)*3, axis=-1)

        if self.show_name:
            img = cv2.putText(img, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.show_bboxes:
            img = cv2.drawContours(img, self.bboxes, -1, (0, 0, 255), 2)

        return img
    
    def toggle_temperature(self): 
        self.show_temperature = not self.show_temperature
        print("SHOW TEMP " + str(self.show_temperature))

    def set_show_temperature(self, val):
        self.show_temperature = val

    def toggle_name(self):
        self.show_name = not self.show_name
    
    def toggle_bboxes(self):
        self.show_bboxes = not self.show_bboxes

    def set_show_name(self,val):
        self.show_name = val

    def set_bboxes(self,val):
        self.show_bboxes = val

    def get_original(self):
        return self.orig_image
    
    def get_original_in_shape(self):
        return self.image
    
    def bottom_left(self):
        return (self.orig_image.shape[0] - 20, 20)
    
    def top_left(self):
        return (50, 50)
    
    def bottom_right(self):
        return (self.orig_image.shape[0] - 20, self.orig_image.shape[1] - 20)
    
    def top_right(self):
        return (50, self.orig_image.shape[1] - 20)
    
class DualImage():

    # vertical offset = 2
    # horizontal offset = 90
    # scale_fact = 2.3

    IR = {
        'TOP_LEFT'  : (10,  50),
        'BOT_LEFT'  : (450, 100),
        'BOT_RIGHT' : (450, 500)
    }

    RGB = {
        'TOP_LEFT'  : (100, 100),
        'BOT_LEFT'  : (3000, 40),
        'BOT_RIGHT' : (3000, 3500)
    }

    def __init__(self, pathRGB, pathIR, temps, shape, bboxes, defects ,scale_fact, voffset, hoffset):
        
        self.shape = shape
        self.pathRGB = pathRGB
        self.pathIR = pathIR

        self.show_name = False
        self.show_bboxes = False
        self.show_temperature = False
        self.show_defect = False
        self.temp = 0

        self.def_text = ""

        self.scale_fact = scale_fact
        self.voffset = voffset
        self.hoffset = hoffset


        rgb_im   = cv2.imread(pathRGB)
        ir_im    = cv2.imread(pathIR)
        rgb_name = pathRGB.split('/')[-1]
        ir_name  = pathIR.split('/')[-1]

        self.name = rgb_name[:-6]

        self.ir_bboxes   = bboxes
        self.rgb_bboxes = self.get_rgb_bboxes(rgb_im.shape, ir_im.shape, bboxes)

        self.rgb   = Image(rgb_im, rgb_name, shape, self.rgb_bboxes)
        self.ir    = Image(ir_im, ir_name, shape, self.ir_bboxes)
        self.main_image = self.ir
        self.act_main = 'IR'
        
        self.temperatures = temps

        self.defects = defects

    def get_rgb_bboxes(self, rgb_shape, ir_shape, bboxes):

        rgb_bboxes = []

        for bbox in bboxes:
            rgb_bbox = []
            for point in bbox:

                h, w   = ir_shape[0] * self.scale_fact, ir_shape[1] * self.scale_fact
                hh, ww = rgb_shape[:2]
                yoff = (hh - h) // 2
                xoff = (ww - w) // 2

                p0 = int(point[0] * self.scale_fact + xoff + self.hoffset)
                p1 = int(point[1] * self.scale_fact + yoff + self.voffset)
                rgb_bbox.append([p0, p1])

            rgb_bboxes.append(np.array(rgb_bbox))

        return rgb_bboxes
    
    def toggle_defects(self):
        self.show_defect = not self.show_defect
        print("SHOW DEFECT " + str(self.show_defect))

    def set_defect(self, x, y):
        self.def_text = "None"
        self.crop     = None
        for defect in self.defects:
            d = np.array([np.array(point) for point in defect['bbox']])
            if cv2.pointPolygonTest(d, (x, y), False) == 1:
                self.def_text = defect['defect']
                self.crop     = defect['bbox']
                break  

    def get_crop(self):
        x,y,w,h = cv2.boundingRect(np.array(self.crop))
        img = self.ir.get_original()[y:y+h, x:x+w, :]
        return img

    def set_temp(self, x, y):
        self.temp = self.temperatures[y, x]
    
    def toggle_temperature(self):
        self.show_temperature = not self.show_temperature
        self.ir.toggle_temperature()
    
    def set_show_temperature(self, val):
        self.show_temperature = val
        self.ir.set_show_temperature(val)
    
    def set_show_name(self, val):
        self.show_name = val
        self.rgb.set_show_name(val)
        self.ir.set_show_name(val)
    
    def set_show_bboxes(self, val):
        self.show_bboxes = val
        self.rgb.set_bboxes(val)
        self.ir.set_bboxes(val)

    def set_scale_fact(self, val):
        self.scale_fact = val
        self.rgb_bboxes = self.get_rgb_bboxes(self.rgb.orig_image.shape, self.ir.orig_image.shape, self.ir_bboxes)

    def set_voffset(self, val):
        self.voffset = val
        self.rgb_bboxes = self.get_rgb_bboxes(self.rgb.orig_image.shape, self.ir.orig_image.shape, self.ir_bboxes)

    def set_hoffset(self, val):
        self.hoffset = val
        self.rgb_bboxes = self.get_rgb_bboxes(self.rgb.orig_image.shape, self.ir.orig_image.shape, self.ir_bboxes)

    def reset_image(self):
        if self.act_main == 'IR':
            im = self.ir.reset_image()
            if self.show_name:
                im = cv2.putText(im, self.ir.name, self.IR['TOP_LEFT'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if self.show_temperature and self.act_main == 'IR':
                im = cv2.putText(im, str(self.temp), self.IR['BOT_LEFT'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if self.show_defect:
                im = cv2.putText(im, self.def_text, self.IR['BOT_RIGHT'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return im
        
        elif self.act_main == 'RGB':
            im = self.rgb.reset_image()
            if self.show_name:
                im = cv2.putText(im, self.rgb.name, self.RGB['TOP_LEFT'], cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20)
            if self.show_defect:
                im = cv2.putText(im, self.def_text, self.RGB['BOT_RIGHT'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return im
        
    def switch_image(self):
        if self.act_main == 'IR':
            self.main_image = self.rgb
            self.act_main = 'RGB'
        else:
            self.act_main = 'IR'
            self.main_image = self.ir

        return self.reset_image()

    def toggle_name(self):
        self.show_name = not self.show_name
        self.rgb.toggle_name()
        self.ir.toggle_name()

    def toggle_bboxes(self):
        self.show_bboxes = not self.show_bboxes
        self.rgb.toggle_bboxes()
        self.ir.toggle_bboxes()

    def filter_temp(self, min_tem, max_tem):
        
        im = np.zeros_like(self.temperatures, np.uint8)
        im[np.logical_and(self.temperatures > min_tem, self.temperatures < max_tem)] = 255
        im = np.stack((im,)*3, axis=-1)
        print(im.shape)
        print(im.dtype)

        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']

        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect and self.act_main == 'IR':
            im = cv2.putText(im, self.def_text,  br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_bboxes and self.act_main == 'IR':
            im = cv2.drawContours(im, self.ir_bboxes, -1, (0, 0, 255), 2)

        return im

    def filter_hue(self, min_hue, max_hue):
        im = self.main_image.filter_hue(min_hue, max_hue)
        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']

        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect and self.act_main == 'IR':
            im = cv2.putText(im, self.def_text,  br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return im

    def filter_sat(self, min_sat, max_sat):
        im = self.main_image.filter_sat(min_sat, max_sat)
        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']

        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect:
            im = cv2.putText(im, self.def_text, br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return im
    
    def filter_val(self, min_val, max_val):
        im = self.main_image.filter_val(min_val, max_val)
        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']
        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect:
            im = cv2.putText(im, self.def_text, br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return im
        
    def filter_image(self, act_values):
        im = self.main_image.filter_image(act_values)
        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']

        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect:
            im = cv2.putText(im, self.def_text, br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return im
    
    def merge_images(self, alpha):
        
        rgb = self.rgb.get_original()
        ir  = self.ir.get_original()

        return self.merge(alpha, rgb, ir)
    
    def merge(self, alpha, rgb, ir):

        ir  = cv2.resize(ir, (int(ir.shape[1]*self.scale_fact), int(ir.shape[0]*self.scale_fact)))

        pad   = np.zeros_like(rgb, np.uint8)

        h, w   = ir.shape[:2]
        hh, ww = rgb.shape[:2]
        yoff = (hh - h) // 2
        xoff = (ww - w) // 2
        
        self.voffset = int(self.voffset)
        self.hoffset = int(self.hoffset)

        pad[yoff + self.voffset: yoff+h+self.voffset, xoff+self.hoffset: xoff+w+self.hoffset, :] = ir

        merged = cv2.addWeighted(rgb, alpha, pad, 1-alpha, 0)

        
        if self.show_bboxes:
            merged = cv2.drawContours(merged, self.rgb_bboxes, -1, (0, 0, 255), 8)

        if self.show_name:
            merged = cv2.putText(merged, self.rgb.name, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20)
    
        return merged

    def merge_with_filters(self, alpha, values):

        rgb = self.rgb.get_original().copy()
        ir  = self.ir.filter_image(values) * 255
        ir  = np.stack((ir,)*3, axis=-1)
        
        return self.merge(alpha, rgb, ir)
    
    def get_thermal_image(self):

        t = np.copy(self.temperatures)
        max_temp = np.max(t)
        min_temp = np.min(t)

        t = (t - min_temp) / (max_temp - min_temp)
        print(np.max(t))
        print(np.min(t))
        print(np.mean(t))
        print(t.dtype)
        t = t * 255
        t = np.uint8(t)

        im = np.stack((t,)*3, axis=-1)

        bl = self.IR['BOT_LEFT'] if self.act_main == 'IR' else self.RGB['BOT_LEFT']
        br = self.IR['BOT_RIGHT'] if self.act_main == 'IR' else self.RGB['BOT_RIGHT']
        if self.show_temperature and self.act_main == 'IR':
            im = cv2.putText(im, str(self.temp), bl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_defect and self.act_main == 'IR':
            im = cv2.putText(im, self.def_text,  br, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.show_bboxes and self.act_main == 'IR':
            im = cv2.drawContours(im, self.ir_bboxes, -1, (0, 0, 255), 2)

        return im