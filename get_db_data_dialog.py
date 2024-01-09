import os.path

import dearpygui.dearpygui as dpg
from libs.odbc_access_lib import expODBC
from libs.dpgfiledialog import dpgDirFileDialog
import numpy as np
from typing import Callable


class GetDBdataDialog:
    def __init__(self, width: int = 800, height: int = 600, callback: Callable | None=None):
        self.width: int = width
        self.height: int = height
        self.db: expODBC | None = None
        self.setup_gui()
        self.callback = callback

    def setup_gui(self) -> None:
        with dpg.stage() as self.id:
            with dpg.child_window(width=self.width, height=self.height):
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Выбрать файл БД', callback=self.btn_callback)
                    self.materials_list = dpg.add_combo(
                        items=[],
                        callback=self.update_experiments_list,
                        width=200,
                    )
                    self.experiments_type = dpg.add_combo(
                        items=['растяжение', 'сжатие'],
                        default_value='растяжение',
                        callback=self.update_experiments_list,
                        width=150,
                    )
                    self.experiments_list = dpg.add_combo(
                        items=[],
                        width=-1,
                        callback=self.on_select_experiment,
                    )
                with dpg.subplots(rows=1, columns=3, width=-1):
                    with dpg.plot(anti_aliased=True):
                        self.pulses_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='время, мс')
                        self.pulses_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='деформация')
                        self.ei_line = dpg.add_line_series(x=[], y=[], label='I', parent=self.pulses_y_axis)
                        self.er_line = dpg.add_line_series(x=[], y=[], label='R', parent=self.pulses_y_axis)
                        self.et_line = dpg.add_line_series(x=[], y=[], label='T', parent=self.pulses_y_axis)
                        self.t1 = dpg.add_drag_line(callback=self.update_lines)
                        self.t2 = dpg.add_drag_line(callback=self.update_lines)
                        dpg.add_plot_legend()
                    with dpg.plot(anti_aliased=True):
                        self.v_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='время, мс')
                        self.v_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='скорость')
                        self.v_line = dpg.add_line_series(x=[], y=[], label='I', parent=self.v_y_axis)
                    with dpg.plot(anti_aliased=True):
                        self.f_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='время, мс')
                        self.f_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='сила')
                        self.f_line = dpg.add_line_series(x=[], y=[], label='I', parent=self.f_y_axis)
                with dpg.table():
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text('радиус, мм')
                        self.r_value = dpg.add_text('')
                        dpg.add_text('длина, мм')
                        self.l_value = dpg.add_text('')
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Прочитать', width=150, callback=self.on_apply, user_data=True)
                    dpg.add_button(label='Отмена', width=150, callback=self.on_apply, user_data=False)
                    dpg.add_button(label='Сохранить csv', width=150, callback=self.save_btn_callback)

    def submit(self, parent) -> None:
        dpg.push_container_stack(parent)
        dpg.unstage(self.id)
        dpg.pop_container_stack()

    def assign_db(self, db_path: str | None):
        if db_path is None:
            return
        if os.path.exists(db_path):
            self.db = expODBC(db_path)
            self.update_materials()

    def btn_callback(self, sender, app_data, user_data):
        fd = dpgDirFileDialog(
            callback=self.assign_db,
            extensions=['accdb'],
        )
        fd.show()

    def update_materials(self):
        if self.db is None:
            return
        dpg.configure_item(
            self.materials_list,
            items=[f'{m['Материал']}-{m['КодМатериала']}' for m in self.db.getMaterials()],
        )
        dpg.set_value(self.materials_list, '')

    def update_experiments_list(self) -> None:
        if self.db is None:
            return
        if not (mat := dpg.get_value(self.materials_list)):
            return
        dpg.set_value(self.experiments_list, '')
        exp_type = {'растяжение': 't', 'сжатие': 'c'}[dpg.get_value(self.experiments_type)]
        mat_code = mat.split('-')[-1]
        numbers = self.db.getNumbers(exp_type, mat_code)
        dpg.configure_item(
            self.experiments_list,
            items=[f'{exp_type}{mat_code}-{n['НомерОбразца']}' for n in numbers],
        )

    def on_select_experiment(self, sender, app_data, user_data):
        if not app_data:
            return
        if self.db is None:
            return
        exp_data = self.db.getExperimentData(app_data)
        t = [tt*1000 for tt in exp_data.pulses['t']]
        ei = exp_data.pulses['pulses'][0]
        er = exp_data.pulses['pulses'][1]
        et = exp_data.pulses['pulses'][2]
        dpg.set_value(self.ei_line, [t, ei])
        dpg.set_value(self.er_line, [t, er])
        dpg.set_value(self.et_line, [t, et])
        dpg.set_value(self.t1, t[0])
        dpg.set_value(self.t2, t[-1])
        dpg.fit_axis_data(self.pulses_x_axis)
        dpg.fit_axis_data(self.pulses_y_axis)
        self.update_lines()
        dpg.set_value(self.r_value, exp_data.d0/2)
        dpg.set_value(self.l_value, exp_data.l0)

    def update_lines(self):
        if self.db is None:
            return
        if not (exp_code := dpg.get_value(self.experiments_list)):
            return
        t = dpg.get_value(self.ei_line)[0]
        n1 = np.searchsorted(t, dpg.get_value(self.t1))
        n2 = np.searchsorted(t, dpg.get_value(self.t2))
        if n1 > n2:
            n1, n2 = n2, n1
        diag = self.db.getDiagram(exp_code)
        if diag is None:
            return
        dpg.set_value(self.v_line, [t[n1:n2], diag['v'][n1:n2]])
        dpg.set_value(self.f_line, [t[n1:n2], diag['F'][n1:n2]])
        dpg.fit_axis_data(self.v_x_axis)
        dpg.fit_axis_data(self.v_y_axis)
        dpg.fit_axis_data(self.f_x_axis)
        dpg.fit_axis_data(self.f_y_axis)

    def on_apply(self, sender, app_data, user_data):
        result = {}
        if user_data and dpg.get_value(self.experiments_list):
            result['r'] = float(dpg.get_value(self.r_value))
            result['l'] = float(dpg.get_value(self.l_value))
            t, v = dpg.get_value(self.v_line)
            t, f = dpg.get_value(self.f_line)
            result['t'] = [(tt-t[0]) for tt in t]
            result['v'] = v
            result['f'] = [-ff for ff in f]
        if self.callback:
            self.callback(result)

    def save_csv(self, file_path: str):
        t, v = dpg.get_value(self.v_line)
        t, f = dpg.get_value(self.f_line)
        if not v:
            return
        if not f:
            return
        with open(file_path, 'w') as fout:
            fout.write('t, v, F\n')
            for i in range(len(t)):
                fout.write(f'{t[i]-t[0]}, {v[i]}, {-f[i]}\n')

    def save_btn_callback(self, sender, app_data, user_data):
        fd = dpgDirFileDialog(save_mode=True, callback=self.save_csv)
        fd.show()




if __name__ == '__main__':
    dpg.create_context()
    dpg.create_viewport(title="DB data")
    dpg.setup_dearpygui()
    with dpg.font_registry():
        with dpg.font("c:/Windows/Fonts/arial.ttf", 16, default_font=True) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
            dpg.bind_font(default_font)
    with dpg.window() as main_window:
        d = GetDBdataDialog()
        d.submit(main_window)
    dpg.show_viewport()
    dpg.set_primary_window(main_window, True)
    dpg.start_dearpygui()
    dpg.destroy_context()
