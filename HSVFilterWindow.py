import cv2
import os
import json
import numpy as np
from defect_visualizer.Image import Image, DualImage
from defect_visualizer.Commands import CommandParser
from thermal import Thermal

class HSVFilterWindow():

    CLIP_PARAM  = {
        'hue'        : [0, 360],
        'saturation' : [0, 255],
        'value'      : [0, 255]
    }

    def __init__(self, win_name, load_path, shape, mode, dual, bbox_file, use_bbox):

        self.command_parser = CommandParser()
        self.num_crops = 0
        self.win_name = win_name

        self.thermal_parser = Thermal(
            dirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libdirp.so',
            dirp_sub_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_dirp.so',
            iirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_iirp.so',
            exif_filename='plugins/exiftool-12.35.exe',
            dtype=np.float32,
        )

        self.scale_fact  = 1
        self.voffset     = 0
        self.hoffset     = 0
        
        with open(bbox_file, 'r') as f:
            self.bbox_file = json.load(f)

        self.window_name = 'Trackbars'
        self.parameters  = ['hue', 'saturation', 'value']
        self.act_values  = self.CLIP_PARAM.copy()

        self.mode = mode if mode == 'dir' or mode == 'img' else None
        assert self.mode is not None , 'Invalid mode, please use img or dir'

        self.dual = dual

        self.shape = shape
        self.show  = None

        self.max_temp = 50
        self.min_temp = 0

        self.alpha = .5

        if mode == 'img':
            self.im_path   = load_path
            if self.dual:
                if load_path.endswith('W.JPG'):
                    dual_path   = load_path[:-5] + 'T.JPG'
                    self.bboxes = [b['bbox'] for b in self.bbox_file[dual_path[:-4]]] if dual_path[:-4] in self.bbox_file.keys() else []
                    bb          = [np.array(p) for p in self.bboxes]
                    temp        = self.thermal_parser.parse_dirp2(dual_path)
                    defects     = self.bbox_file[dual_path[:-4]] if dual_path[:-4] in self.bbox_file.keys() else []
                    self.image  = DualImage(self.im_path, dual_path, temp, self.shape, bb, defects, self.scale_fact, self.voffset, self.hoffset)
                elif load_path.endswith('T.JPG'):
                    dual_path   = load_path[:-5] + 'W.JPG'
                    self.bboxes = [b['bbox'] for b in self.bbox_file[load_path[:-4]]] if load_path[:-4] in self.bbox_file.keys() else []
                    bb          = [np.array(p) for p in self.bboxes]
                    temp        = self.thermal_parser.parse_dirp2(load_path)
                    defects     = self.bbox_file[load_path[:-4]] if load_path[:-4] in self.bbox_file.keys() else []
                    self.image  = DualImage(dual_path, self.im_path, self.shape, bb, defects, self.scale_fact, self.voffset, self.hoffset)
            else:
                if use_bbox:
                    self.bboxes    = [b['bbox'] for b in self.bbox_file[load_path[:-4]]]
                else:
                    self.bboxes    = []

                crop = cv2.imread(load_path)
                self.image     = Image(crop, 'crop', self.shape, self.bboxes)
                

        elif mode == 'dir':
            self.image_dir_path = load_path
            if self.dual:
                self.image_list = self.get_dual_image_list(load_path)
                print("Found " + str(len(self.image_list)) + " images")
                self.idx        = 0
                self.im_path    = self.image_list[self.idx]
                name            = self.im_path[1].split('/')[-1][:-4]
                self.bboxes     = [b['bbox'] for b in self.bbox_file[name]] if name in self.bbox_file.keys() else []
                bb              = [np.array(p) for p in self.bboxes]
                temp            = self.thermal_parser.parse_dirp2(self.image_list[self.idx][1])
                defects         = self.bbox_file[name] if name in self.bbox_file.keys() else []
                self.image      = DualImage(self.image_list[self.idx][0], self.image_list[self.idx][1], temp, self.shape, bb, defects,self.scale_fact, self.voffset, self.hoffset)
            else:
                self.image_list     = os.listdir(load_path)
                print("Found " + str(len(self.image_list)) + " images")
                self.idx            = 0
                self.im_path        = os.path.join(load_path, self.image_list[self.idx])
                self.bboxes         = [b['bbox'] for b in self.bbox_file[self.im_path[:-4]]]  if self.im_path[:-4] in self.bbox_file.keys() else []
                self.image          = Image(self.im_path, self.shape, [np.array(p) for p in self.bboxes])


        self.show         = self.image.reset_image()
        self.last_command = 'r'

    def get_dual_image_list(self, path):

        used = []
        dual_images = []

        allowed_ext = ['.jpg', '.png', '.jpeg']

        for f in os.listdir(path):

            if f not in used:
                for ext in allowed_ext:
                    if f.lower().endswith(ext):
                        print("Found image " + f)
                        used.append(f)
                        
                        if f.endswith('_W.JPG'):
                            if f[:-5] + 'IR.JPG' not in used and os.path.isfile(os.path.join(path,f[:-5] + 'IR.JPG')):
                                print("Found dual image " + f[:-5] + 'IR.JPG')
                                dual_images.append((os.path.join(path, f), os.path.join(path, f[:-5] + 'IR.JPG')))
                                used.append(f[:-5] + 'IR.JPG')

                        elif f.endswith('_T.JPG'):
                            if f[:-5] + 'W.JPG' not in used and os.path.isfile(os.path.join(path,f[:-5] + 'W.JPG')):
                                print("Found dual image " + f[:-5] + '_W.JPG')
                                dual_images.append((os.path.join(path, f[:-5] + 'W.JPG'), os.path.join(path, f)))
                                used.append(f[:-5] + '_W.JPG')

                        break
        
        return dual_images
    
    def toggle_name(self):
        self.image.toggle_name()

    def toggle_bboxes(self):
        self.image.toggle_bboxes()

    def toggle_temperature(self):
        self.image.toggle_temperature()

    def toggle_defects(self):
        self.image.toggle_defects()

    def reset_trackbars(self):
            
        for param in self.parameters:
            cv2.setTrackbarPos('Max ' + param, self.win_name, self.CLIP_PARAM[param][1])
            cv2.setTrackbarPos('Min ' + param, self.win_name, self.CLIP_PARAM[param][0])

        self.alpha = .5
        cv2.setTrackbarPos('Alpha', self.win_name, int(self.alpha * 100))
        cv2.setTrackbarPos('Max temp', self.win_name, 50)
        cv2.setTrackbarPos('Min temp', self.win_name, 0)
        self.act_values = self.CLIP_PARAM.copy()

    def set_trackbars(self):

        for param in self.parameters:
            cv2.setTrackbarPos('Max ' + param, self.win_name, self.act_values[param][1])
            cv2.setTrackbarPos('Min ' + param, self.win_name, self.act_values[param][0])

        cv2.setTrackbarPos('Alpha', self.win_name, int(self.alpha * 100))

        cv2.setTrackbarPos('Max temp', self.win_name, self.max_temp)
        cv2.setTrackbarPos('Min temp', self.win_name, self.min_temp)

    def load_config(self, config_path):

        with open(config_path, 'r') as f:
            config = json.load(f)

        for param in self.parameters:
            self.act_values[param] = config[param]
        
        if config['dual']:
            self.im_path    = (config['image_path'][0], config['image_path'][1])
            self.shape      = config['shape']
            self.bboxes     = config['bboxes']
            self.hoffset    = config['hoffset']
            self.voffset    = config['voffset']
            self.scale_fact = config['scale_fact']
            self.alpha      = config['alpha']
            bb              = [np.array(p) for p in self.bboxes]
            temp            = self.thermal_parser.parse_dirp2(config['image_path'][1])
            defects         = self.image.defects
            self.image      = DualImage(self.im_path[0], self.im_path[1], temp, self.shape, bb, defects, self.scale_fact, self.voffset, self.hoffset)
            self.image.set_show_name(config['show_name'])
            self.image.set_show_bboxes(config['show_bboxes'])
            self.show       = self.image.reset_image()
            self.mode       = 'img',
            self.max_temp   = config['max_temp']
            self.min_temp   = config['min_temp']
        else:
            self.im_path    = config['image_path']
            self.shape      = config['shape']
            self.bboxes     = config['bboxes']
            self.image      = Image(self.im_path, self.shape, self.bboxes)
            self.show       = self.image.reset_image()
            self.mode = 'img'

        command = config['command']

        if command == 'a':
            self.show = self.image.filter_image(self.act_values)
        elif command == 'h':
            self.show = self.image.filter_hue(self.act_values['hue'][0], self.act_values['hue'][1])
        elif command == 's':
            self.show = self.image.filter_sat(self.act_values['saturation'][0], self.act_values['saturation'][1])
        elif command == 'v':
            self.show = self.image.filter_val(self.act_values['value'][0], self.act_values['value'][1])
        elif command == 'r':
            self.show = self.image.reset_image()
        elif command == 'm':
            self.show = self.image.merge_images(self.alpha)
        elif command == 'z':
            self.show = self.image.filter_temp(self.min_temp, self.max_temp)

        print('Loaded config from ' + config_path)
        print(config)

    def save_config(self, config_path):

        config = {
            'image_path' : self.im_path,
            'command'    : self.last_command,
            'hue'        : self.act_values['hue'],
            'saturation' : self.act_values['saturation'],
            'value'      : self.act_values['value'],
            'shape'      : self.shape,
            'dual'       : self.dual,
            'alpha'      : self.alpha,
            'bboxes'     : self.bboxes,
            'scale_fact' : self.scale_fact,
            'voffset'    : self.voffset,
            'hoffset'    : self.hoffset,
            'show_narme'  : self.image.show_name,
            'show_bboxes': self.image.show_bboxes,
            'defects'    : self.image.defects,
            'max_temp'   : self.max_temp,
            'min_temp'   : self.min_temp
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent  = 4)

        print('Saved config at ' + config_path)

    def next_image(self):
        if self.mode == 'dir':
            self.idx = (self.idx + 1) % len(self.image_list)
            if self.dual:
                self.im_path = (self.image_list[self.idx][0],  self.image_list[self.idx][1])
                name = self.im_path[1].split('/')[-1][:-4] 
                self.bboxes = [b['bbox'] for b in self.bbox_file[name]] if name in self.bbox_file.keys() else []
                bb          = [np.array(p) for p in self.bboxes]
                rgb_path    = self.image_list[self.idx][0]
                ir_path     = self.image_list[self.idx][1]
                temp        = self.thermal_parser.parse_dirp2(ir_path)
                defects     = self.bbox_file[name] if name in self.bbox_file.keys() else []
                self.image  = DualImage(rgb_path, ir_path, temp, self.shape, bb, defects, self.scale_fact, self.voffset, self.hoffset)
            else:
                self.im_path = os.path.join(self.image_dir_path, self.image_list[self.idx])
                name  = self.im_path.split('/')[-1][:-4]
                self.bboxes = [b['bbox'] for b in self.bbox_file[name]] if name in self.bbox_file.keys() else []
                bb          = [np.array(p) for p in self.bboxes]
                self.image = Image(self.im_path, self.shape, bb)
            
            self.show  = self.image.reset_image()

            self.reset_trackbars()
        else:
            print('Not in dir mode')

    def previous_image(self):

        if self.mode == 'dir':
            self.idx = len(self.image_list) - 1 if self.idx == 0 else self.idx - 1
            if self.dual:
                self.im_path = (self.image_list[self.idx][0],  self.image_list[self.idx][1])
                name = self.im_path[1].split('/')[-1][:-4] 
                self.bboxes = [b['bbox'] for b in self.bbox_file[name]] if name in self.bbox_file.keys() else []
                bb          = [np.array(p) for p in self.bboxes]
                rgb         = self.image_list[self.idx][0]
                ir          = self.image_list[self.idx][1]
                temp        = self.thermal_parser.parse_dirp2(ir)
                defects     = self.bbox_file[name] if name in self.bbox_file.keys() else []
                self.image  = DualImage(rgb, ir, temp, self.shape, bb, defects, self.scale_fact, self.voffset, self.hoffset)
            else:
                self.im_path = os.path.join(self.image_dir_path, self.image_list[self.idx])
                name  = self.im_path.split('/')[-1][:-4]
                self.bboxes = [b['bbox'] for b in self.bbox_file[name]]  if name in self.bbox_file.keys() else []
                bb          = [np.array(p) for p in self.bboxes]
                self.image = Image(self.im_path, self.shape, bb)
            
            self.show  = self.image.reset_image()

            self.reset_trackbars()
        else:
            print("Not in dir mode")

    def apply_hsv_filter(self):
        self.show = self.image.filter_image(self.act_values)
        self.last_command = 'a'

    def apply_hue_filter(self):
        self.show = self.image.filter_hue(self.act_values['hue'][0], self.act_values['hue'][1])
        self.last_command = 'h'

    def apply_sat_filter(self):
        self.show = self.image.filter_sat(self.act_values['saturation'][0], self.act_values['saturation'][1])
        self.last_command = 's'

    def apply_val_filter(self):
        self.show = self.image.filter_val(self.act_values['value'][0], self.act_values['value'][1])
        self.last_command = 'v'

    def merge(self, scale_fact, voffset, hoffset):
        if self.dual:
            self.scale_fact = scale_fact
            self.voffset    = voffset
            self.hoffset    = hoffset
            self.image.set_voffset(voffset)
            self.image.set_hoffset(hoffset)
            self.image.set_scale_fact(scale_fact)
            self.show = self.image.merge_images(self.alpha)
            self.last_command = 'm'
        else:
            print("Not in dual mode")

    def merge_with_filters(self):
        if self.dual:
            self.scale_fact  = float(input('Enter the scale factor: '))
            self.voffset     = int(input('Enter the vertical offset: '))
            self.hoffset     = int(input('Enter the horizontal offset: '))
            self.show        = self.image.merge_with_filters(self.alpha, self.act_values)
        else:
            print("Not in dual mode")
    
    def toggle_dual(self):
        if self.dual:
            self.show = self.image.switch_image()
        else:
            print("Not in dual mode")

    def apply_temp_filter(self):
        self.show = self.image.filter_temp(self.min_temp, self.max_temp)
        self.last_command = 't'

    def show_thermal_image(self):
        self.show = self.image.get_thermal_image()

    def show_window(self):

        cv2.namedWindow(self.win_name)
        cv2.resizeWindow(self.win_name, self.shape[0], self.shape[1])
        cv2.setMouseCallback(self.win_name, self.draw_temperature)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('Max hue', self.win_name, self.CLIP_PARAM['hue'][0], self.CLIP_PARAM['hue'][1], self.change_max_hue)
        cv2.createTrackbar('Min hue', self.win_name, self.CLIP_PARAM['hue'][0], self.CLIP_PARAM['hue'][1], self.change_min_hue)

        cv2.createTrackbar('Max saturation', self.win_name, self.CLIP_PARAM['saturation'][0], self.CLIP_PARAM['saturation'][1], self.change_max_saturation)
        cv2.createTrackbar('Min saturation', self.win_name, self.CLIP_PARAM['saturation'][0], self.CLIP_PARAM['saturation'][1], self.change_min_saturation)

        cv2.createTrackbar('Max value', self.win_name, self.CLIP_PARAM['value'][0], self.CLIP_PARAM['value'][1], self.change_max_value)
        cv2.createTrackbar('Min value', self.win_name, self.CLIP_PARAM['value'][0], self.CLIP_PARAM['value'][1], self.change_min_value)

        cv2.createTrackbar('Max temp', self.win_name, 0, 50, self.change_max_temp)
        cv2.createTrackbar('Min temp', self.win_name, 0, 50, self.change_min_temp)

        cv2.createTrackbar('Alpha', self.win_name, 0, 100, self.change_alpha)
        cv2.imshow(self.win_name, self.show)

        self.set_trackbars()
        
        k = None
        while k != ord('f'):
            cv2.resizeWindow(self.win_name, self.shape[0], self.shape[1])
            cv2.imshow(self.win_name, self.show)
            self.set_trackbars()
            k = cv2.waitKey(0)
            print("COMMAND " + chr(k))
            command = self.command_parser.parse(chr(k))
            if command is not None:
                command.execute(self)

        cv2.destroyAllWindows()

    def draw_temperature(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image.set_temp(x, y)
            self.image.set_defect(x,y)
            print(self.image.temp)
            print(self.image.def_text)

    def crop_pannel(self):
        crop = self.image.get_crop()
        cv2.imwrite('tmp/cropped.jpg', crop)
        cv2.imshow('Cropped' + str(self.num_crops), crop)
        self.num_crops += 1

    def change_alpha(self, val):
        self.alpha = val / 100

    def change_min_hue(self, val):
        
        self.act_values['hue'][0] = val

    def change_max_hue(self, val):
        
        self.act_values['hue'][1] = val

    def change_min_saturation(self, val):
            
        self.act_values['saturation'][0] = val

    def change_max_saturation(self, val):

        self.act_values['saturation'][1] = val

    def change_min_value(self, val):

        self.act_values['value'][0] = val

    def change_max_value(self, val):

        self.act_values['value'][1] = val

    def change_max_temp(self, val):
        self.max_temp = val

    def change_min_temp(self, val):
        self.min_temp = val
