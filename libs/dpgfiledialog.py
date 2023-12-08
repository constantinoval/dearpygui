import dearpygui.dearpygui as dpg
import os

class dpgDirFileDialog:
    def __init__(self, font=None, current_path=None, callback=None, extensions=[],
                 width=500, height=420, dir_mode=False):
        self.dir_mode = dir_mode
        self.width = width
        self.height = height
        self.current_path = os.curdir if (current_path is None or not os.path.exists(current_path)) else current_path
        self.current_path = os.path.abspath(self.current_path)
        self.font = font
        self.selected_file = None
        self.extensions = [e.upper() for e in extensions]
        self.callback = callback
        
    def update_file_list(self, sender, app_data, user_data):
        if not os.path.exists(user_data):
            return
        dirs = []
        files = []
        for f in os.listdir(user_data):
            full_path = os.path.join(user_data, f)
            if os.path.isdir(full_path):
                dirs.append(f)
            elif not self.dir_mode:
                if self.extensions:
                    if f.split('.')[-1].upper() in self.extensions:
                        files.append(f)
                else:
                    files.append(f)
        all_list = dirs + files
        dpg.configure_item('file list', items=all_list)
        self.current_path = os.path.abspath(user_data)

    def dir_back(self):
        lst = self.current_path.split(os.path.sep)
        if len(lst) < 2:
            return
        self.update_file_list(None, None, os.path.sep.join(lst[:-1])+os.path.sep)

    def file_list_callback(self, sender, app_data):
        if os.path.isdir(os.path.join(self.current_path, app_data)):
            self.update_file_list(None, None, os.path.join(self.current_path, app_data))
            dpg.set_value('current file', value=self.current_path)
        else:
            dpg.set_value('current file', value=os.path.join(self.current_path, app_data))
        
        
    def apply_result(self, sender, app_data, user_data):
        f = dpg.get_value('current file')
        self.selected_file = None
        if user_data == 'OK':
            if not self.dir_mode and os.path.isfile(f):
                self.selected_file = f
            if self.dir_mode and os.path.isdir(f):
                self.selected_file = f
        dpg.delete_item('File selection dialog')
        if self.callback:
            self.callback(self.selected_file)

    def show(self):
        drives = (d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f'{d}:'))
        # cur_file_list = os.listdir()
        with dpg.window(label="File selection dialog", width=self.width,
                        height=self.height, tag='File selection dialog', modal=True, no_title_bar=True):
            dpg.add_separator()
            dpg.add_text('Выберите файл:')
            dpg.add_text('', tag='current file')
            dpg.set_value('current file', self.current_path)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text('Диски: ')
                for d in drives:
                    # print(d)
                    dpg.add_button(label=d.upper(), callback=self.update_file_list, user_data=f'{d}:/')
                dpg.add_button(label='..', callback=self.dir_back)
            dpg.add_separator()
            dpg.add_listbox(tag='file list', items=[], width=-1, num_items=15, callback=self.file_list_callback)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=250)
                dpg.add_button(label='OK', callback=self.apply_result, user_data='OK', width=100)
                dpg.add_button(label='Cancel', callback=self.apply_result, user_data='CANCEL', width=100)
            dpg.add_separator()
            if self.font is not None:
                dpg.bind_font(self.font)
        self.update_file_list(None, None, self.current_path)


if __name__=='__main__':
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()
    
    with dpg.window(label="Example Window", width=800, height=600, tag='main'):
        pass
    fd = dpgDirFileDialog(callback=lambda filename: print(filename), dir_mode=True)
    fd.show()
    
    dpg.show_viewport()
    dpg.set_primary_window('main', True)
    dpg.start_dearpygui()
    dpg.destroy_context()