
class Command:

    def __init__(self, key):
        self.key = key

    def execute(self, window):
        pass

    def matches_key(self, k2):
        return ord(self.key) == k2
    
class CommandParser():

    def __init__(self):
        self.commands = {}
        self.commands['l'] = LoadPressetCommand('l')
        self.commands['w'] = SavePressetCommand('w')
        self.commands['r'] = ResetCommand('r')
        self.commands['n'] = NextCommand('n')
        self.commands['p'] = PreviousCommand('p')
        self.commands['a'] = HSVFilterCommand('a')
        self.commands['h'] = HueFilter('h')
        self.commands['s'] = SatFilter('s')
        self.commands['v'] = ValFilter('v')
        self.commands['m'] = MergeCommand('m')
        self.commands['u'] = MergeWithFiltersCommand('u')
        self.commands['d'] = ToggleDualCommand('d')
        self.commands['o'] = ShowNameCommand('o')
        self.commands['q'] = ToggleBboxCommand('q')
        self.commands['t'] = ToggleTemperatureCommand('t')
        self.commands['y'] = ToggleDefectCommand('y')
        self.commands['z'] = FilterTempCommand('z')
        self.commands['x'] = CropPannel('x')
        self.commands['j'] = ShowThermalImageCommand('j')

    def add_command(self, command):
        self.commands[command.key] = command

    def parse(self, key):
        if key in self.commands.keys():
            return self.commands[key]
        else:
            print("Wrong command received")

class LoadPressetCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        path_cfg = input('Enter the name of the preset file to load: ')
        window.load_config(path_cfg)

class SavePressetCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        path_cfg = input('Enter the name of the preset file to save: ')
        window.save_config(path_cfg)


class ResetCommand(Command):

    def __init__(sefl, key):
        super().__init__(key)

    def execute(self, window):
        window.show = window.image.reset_image()
        window.reset_trackbars()


class NextCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.next_image() 

class PreviousCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.previous_image()

class HSVFilterCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.apply_hsv_filter()

class HueFilter(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window): 
        window.apply_hue_filter()

class SatFilter(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.apply_sat_filter()

class ValFilter(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.apply_val_filter()

class MergeCommand(Command):

    def __init__(self, key):
        super().__init__(key)
        self.hoffset = 0
        self.voffset = 0
        self.scale_fact = 1.0

    def execute(self, window):
        self.scale_fact  = float(input('Enter the scale factor: '))
        self.voffset     = int(input('Enter the vertical offset: '))
        self.hoffset     = int(input('Enter the horizontal offset: '))
        window.merge(self.scale_fact, self.voffset, self.hoffset)

class MergeWithFiltersCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.merge_with_filters()

class ToggleDualCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.toggle_dual()

class ShowNameCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.toggle_name()
 
class ToggleBboxCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.toggle_bboxes()

class MergeWithParamsCommand(Command):

    def __init__(self, key, scale_fact, voffset, hoffset):
        super().__init__(key)
        self.scale_fact = scale_fact
        self.voffset = voffset
        self.hoffset = hoffset

    def execute(self, window):
        window.merge(self.scale_fact, self.voffset, self.hoffset)

class ToggleTemperatureCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.toggle_temperature()

class ToggleDefectCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.toggle_defects() 

class FilterTempCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.apply_temp_filter()

class CropPannel(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.crop_pannel()

class ShowThermalImageCommand(Command):

    def __init__(self, key):
        super().__init__(key)

    def execute(self, window):
        window.show_thermal_image()