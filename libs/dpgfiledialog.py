import dearpygui.dearpygui as dpg
import os
from dearpygui import demo




class dpgDirFileDialog:
    def __init__(self, font=None, current_path=None, callback=None, extensions=[],
                 width=500, height=420, dir_mode=False, save_mode=False):
        self.save_mode = save_mode
        self.dir_mode = dir_mode
        self.width = width
        self.height = height
        self.current_path = os.curdir if (current_path is None or not os.path.exists(current_path)) else current_path
        self.current_path = os.path.abspath(self.current_path)
        self.font = font
        self.selected_path = None
        self.extensions = [e.upper() for e in extensions]
        self.callback = callback        
        with dpg.theme() as self.table_theme:
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 126, 168), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
        self.selected_item = -1

    def full_path(self, local_file: str):
        return os.path.join(self.current_path, local_file)

    def get_file_list(self):
        if not os.path.exists(self.current_path):
            return
        dirs = []
        files = []
        for f in os.listdir(self.current_path):
            if os.path.isdir(self.full_path(f)):
                dirs.append(f)
            elif not self.dir_mode:
                if self.extensions:
                    if f.split('.')[-1].upper() in self.extensions:
                        files.append((f, os.path.getsize(self.full_path(f))))
                else:
                    files.append((f, os.path.getsize(self.full_path(f))))
        return dirs, files

    def select_item(self, sender, app_data, user_data):
        if self.selected_item != -1 and self.selected_item != user_data[0]:
            dpg.unhighlight_table_row(self.path_table, self.selected_item)
        if user_data[0] != self.selected_item:
            dpg.highlight_table_row(self.path_table, user_data[0], color=(0, 164, 230))
        self.selected_item = user_data[0]
        self.selected_file = user_data[1]
        try:
            dpg.set_value('current file', self.selected_file)
        except:
            pass
        if self.save_mode and os.path.isfile(self.selected_file):
            dpg.set_value(self.new_file_name, self.selected_file.split(os.path.sep)[-1])

    def update_file_list(self, sender, app_data, user_data=None):
        if user_data:
            self.current_path = user_data
            self.selected_file = user_data
            self.selected_item = -1
        dpg.delete_item(self.path_table, children_only=True)
        dpg.add_table_column(init_width_or_weight=0.2, label='type', parent=self.path_table)
        dpg.add_table_column(init_width_or_weight=0.5, label='name', parent=self.path_table)
        dpg.add_table_column(init_width_or_weight=0.3, label='size', parent=self.path_table)       
        dirs, files = self.get_file_list()
        i = 0
        for d in dirs:
            ud = (i, self.full_path(d))
            with dpg.table_row(parent=self.path_table):
                dpg.add_selectable(label='[dir]', span_columns=True, user_data=ud, callback=self.select_item)
                dpg.bind_item_handler_registry(dpg.last_item(), "file_dialog_double_click")
                dpg.add_selectable(label=d, span_columns=True, user_data=ud, callback=self.select_item)
                dpg.add_selectable(label='', span_columns=True, user_data=ud, callback=self.select_item)
            i += 1
        for f, s in files:
            ud = (i, self.full_path(f))
            with dpg.table_row(parent=self.path_table):
                dpg.add_selectable(label='', span_columns=True, user_data=ud, callback=self.select_item)
                dpg.bind_item_handler_registry(dpg.last_item(), "file_dialog_double_click")
                dpg.add_selectable(label=f, span_columns=True, user_data=ud, callback=self.select_item)
                dpg.add_selectable(label=s, span_columns=True, user_data=ud, callback=self.select_item)
            i += 1
        dpg.set_value('current file', self.current_path)

    def dir_back(self):
        lst = self.current_path.split(os.path.sep)
        while '' in lst:
            lst.remove('')
        if len(lst) < 1:
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
            if self.save_mode:
                self.selected_file = os.path.join(self.current_path, dpg.get_value(self.new_file_name))
            else:
                if not self.dir_mode and os.path.isfile(f):
                    self.selected_file = f
                if self.dir_mode and os.path.isdir(f):
                    self.selected_file = f
        if self.callback:
            self.callback(self.selected_file)
        dpg.delete_item('file_dialog_double_click')
        dpg.delete_item('File selection dialog')
            
    def double_click_callback(self, sender, app_data, user_data):
        btn = app_data[0]
        if btn!=0:
            return
        item_id = app_data[1]
        clicked_data: str = dpg.get_item_user_data(item_id)[1]
        if os.path.isdir(clicked_data):
            self.update_file_list(None, None, clicked_data)
        else:
            self.apply_result(None, None, 'OK')

    def new_dir_callback(self, sender, app_data, user_data):
        dpg.configure_item('new_dir_group', show=False)
        new_dir_name = dpg.get_value(self.new_dir_name)
        if new_dir_name:
            os.mkdir(self.full_path(new_dir_name))
            self.update_file_list(None, None, None)
            dpg.set_value(self.new_dir_name, '')  
    
    def cancel_new_dir_callback(self, sendedr, app_data, user_data):
        dpg.set_value(self.new_dir_name, '')  
        dpg.configure_item('new_dir_group', show=False)

    def show(self):
        drives = (d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f'{d}:'))
        # cur_file_list = os.listdir()
        with dpg.window(label="File selection dialog", width=self.width,
                        height=self.height, tag='File selection dialog', modal=True, no_title_bar=True):
            dpg.add_separator()
            dpg.add_text('Выберите директорию:' if self.dir_mode else 'Выберите файл:')
            dpg.add_text(self.current_path, tag='current file')
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text('Диски: ')
                for d in drives:
                    # print(d)
                    dpg.add_button(label=d.upper(), callback=self.update_file_list, user_data=f'{d}:'+os.path.sep)
                dpg.add_button(label='..', callback=self.dir_back)
                with dpg.tooltip(parent=dpg.last_item()):
                    dpg.add_text('На уровень вверх')
                dpg.add_button(label='+', callback=lambda: dpg.configure_item('new_dir_group', show=True))
                with dpg.tooltip(parent=dpg.last_item()):
                    dpg.add_text('Создать новый каталог')
            dpg.add_separator()
            with dpg.group(horizontal=True, tag='new_dir_group', show=False):
                self.new_dir_name = dpg.add_input_text(hint='введите имя')
                dpg.add_button(label='OK', callback=self.new_dir_callback)
                dpg.add_button(label=' X ', callback=self.cancel_new_dir_callback)
            dpg.add_separator()
            with dpg.table(
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                row_background=True,
                borders_outerH=True,
                borders_outerV=True,
                borders_innerH=True,
                borders_innerV=True,
                height=-100 if self.save_mode else -80,
                scrollY=True,
                ) as self.path_table:
                pass
            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_separator()
            if self.save_mode:
                self.new_file_name = dpg.add_input_text(hint='имя файла', width=-1)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label='OK', callback=self.apply_result, user_data='OK', width=100)
                dpg.add_button(label='Cancel', callback=self.apply_result, user_data='CANCEL', width=100)
            dpg.add_separator()
            if self.font is not None:
                dpg.bind_font(self.font)
        dpg.bind_item_theme(self.path_table, self.table_theme)
        with dpg.item_handler_registry(tag="file_dialog_double_click"):
            dpg.add_item_double_clicked_handler(callback=self.double_click_callback)
        self.update_file_list(None, None, self.current_path)



if __name__=='__main__':
    dpg.create_context()
    dpg.create_viewport(width=500)
    dpg.setup_dearpygui()
    with dpg.font_registry():
        with dpg.font(file="./XO_Caliburn_Nu.ttf", size=18) as font1:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    dpg.bind_font(font1)
    
    with dpg.window(label="Example Window", width=800, height=900, tag='main'):
        fd = dpgDirFileDialog(callback=lambda filename: print(filename), dir_mode=False, save_mode=False)
        fd.show()
    
    dpg.show_viewport()
    # demo.show_demo()
    dpg.start_dearpygui()
    dpg.destroy_context()