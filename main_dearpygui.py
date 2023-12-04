import dearpygui.dearpygui as dpg
from enum import StrEnum
from libs.dpgfiledialog import dpgDirFileDialog
from libs.lsmesh_lib import boundBox, lsdyna_model
import os


class DPG_WIDGETS(StrEnum):
    MESH_DRAW_BACKGROUND = "MESH_DRAW_BACKGROUND"
    MESH_DRAW_LAYER = "MESH_DRAW_LAYER"
    MESH_DRAW_AREA = "MESH_DRAW_AREA"
    MODEL_PATH_TEXT = "MODEL_PATH_TEXT" 
    MODEL_CHOOSE_DIALOG_BTN = "MODEL_CHOOSE_DIALOG_BTN" 
    MODEL_INFO_TEXT = "MODEL_INFO_TEXT"
    EXP_F_PLOT = "EXP_F_PLOT"
    EXP_V_PLOT = "EXP_V_PLOT"
    EXP_DATA_PATH = "EXP_DATA_PATH"
    PLOT_EXP_V_X = "PLOT_EXP_V_X"
    PLOT_EXP_V_Y = "PLOT_EXP_V_Y"
    PLOT_EXP_F_X = "PLOT_EXP_F_X"
    PLOT_EXP_F_Y = "PLOT_EXP_F_Y"
    EXP_V_LINE = "EXP_V_LINE"
    EXP_F_LINE = "EXP_F_LINE"


MESH_DRAW_AREA_WIDTH = 300
MESH_DRAW_AREA_HEIGH = 300


# Main class
class TrueDiagrammCalculator:
    def __init__(self) -> None:
        self.model_path: str = ""
        self.solver_path: str = ""
        self.working_dir: str = os.path.join(os.path.abspath(os.curdir), "solution")
        self.model: lsdyna_model | None = None
        self.model_bbox: boundBox | None = None

    def assign_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            print(f'Чтение сеточной модели из файла {model_path}...')
            self.model_path = model_path
            self.model = lsdyna_model(model_path, procIncludes=True)
            self.model_bbox = self.model.bbox
            self.model.proceed_node_sets()
            self.draw_mesh(DPG_WIDGETS.MESH_DRAW_LAYER)

    def draw_mesh(self, dpg_draw_layer_tag: str) -> None:
        k = 0.9 * min(MESH_DRAW_AREA_WIDTH / self.model_bbox.a, MESH_DRAW_AREA_HEIGH / self.model_bbox.b)
        x_draw = lambda x: 0.05 * MESH_DRAW_AREA_WIDTH + k * (x - self.model_bbox.xmin)
        y_draw = lambda y: 0.95 * MESH_DRAW_AREA_HEIGH - k * (y - self.model_bbox.ymin)
        dpg.delete_item(dpg_draw_layer_tag, children_only=True)
        for part in self.model.shells:
            for sh in self.model.shells[part].values():
                pnts = [
                    (x_draw(self.model.nodes[n].x), y_draw(self.model.nodes[n].y)) for n in sh.nodes
                ]
                pnts.append(pnts[0])
                dpg.draw_polygon(pnts, fill=(255, 0, 0), color=(0, 0, 0), thickness=1, parent=dpg_draw_layer_tag)
        if 1 in self.model.nodesets:
            for n in self.model.nodesets[1]:
                dpg.draw_circle(
                    center=(
                        x_draw(self.model.nodes[n].x),
                        y_draw(self.model.nodes[n].y),
                    ),
                    radius=3,
                    fill=(0, 255, 0),
                    color=(0, 0, 0),
                    thickness=1,
                    parent=dpg_draw_layer_tag
                )
        if 2 in self.model.nodesets:
            for n in self.model.nodesets[2]:
                x = x_draw(self.model.nodes[n].x)
                y = y_draw(self.model.nodes[n].y)
                dpg.draw_arrow(
                    p1 = (x, y-10),
                    p2 = (x, y),
                    color=(0, 0, 255),
                    thickness=1,
                    parent=dpg_draw_layer_tag
                )

calculator = TrueDiagrammCalculator()


# DPG widgets callbacks
def set_model_text_callback(file_name):
    if file_name:
        dpg.set_value(DPG_WIDGETS.MODEL_PATH_TEXT, file_name)
        calculator.assign_model(file_name)
        model_info = repr(calculator.model) + repr(calculator.model_bbox)
        dpg.set_value(DPG_WIDGETS.MODEL_INFO_TEXT, model_info)


def chose_model_btn_callback(sender, app_data, user_data):
    fd = dpgDirFileDialog(extensions=['k', 'dyn'], height=510, callback=set_model_text_callback)
    fd.show()


def chose_expdata_btn_callback(sender, app_data, user_data):
    fd = dpgDirFileDialog(extensions=['txt', 'csv'], height=510, callback=load_experimental_curves)
    fd.show()


def load_experimental_curves(file_name):
    if file_name:
        dpg.set_value(DPG_WIDGETS.EXP_DATA_PATH, file_name)
# end DPG widgets callbasks

dpg.create_context()
# Настройка кириллического шрифта
with dpg.font_registry():
    with dpg.font(file="./XO_Caliburn_Nu.ttf", size=18) as font1:
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
dpg.bind_font(font1)
dpg.create_viewport(title="Numerical true stress strain diagramm determination", width=800)
dpg.setup_dearpygui()

with dpg.window(label="Example Window", width=600, height=600, tag='main'):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label='Save')
    with dpg.collapsing_header(label="Базовая модель"):
        with dpg.child_window(height=490):
            with dpg.child_window(height=130):
                dpg.add_text("""Описание требований к расчетной модели""")
            with dpg.child_window(height=40):
                with dpg.group(horizontal=True):
                    dpg.add_text("Путь к файлу модели:")
                    dpg.add_input_text(readonly=True, tag=DPG_WIDGETS.MODEL_PATH_TEXT, width=-100)
                    dpg.add_button(label="Открыть", width=100, callback=chose_model_btn_callback)
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=MESH_DRAW_AREA_WIDTH, height=MESH_DRAW_AREA_HEIGH):
                    with dpg.draw_layer(tag=DPG_WIDGETS.MESH_DRAW_BACKGROUND):
                        dpg.draw_rectangle(
                            pmin=(0, 0),
                            pmax=(MESH_DRAW_AREA_WIDTH, MESH_DRAW_AREA_HEIGH),
                            fill=(255, 255, 255)
                        )
                    with dpg.draw_layer(tag=DPG_WIDGETS.MESH_DRAW_LAYER):
                        pass
                with dpg.child_window():
                    dpg.add_text("", tag=DPG_WIDGETS.MODEL_INFO_TEXT)
    with dpg.collapsing_header(label='Экспериментальные кривые'):
        pass


dpg.show_viewport()
dpg.set_primary_window('main', True)
dpg.start_dearpygui()
dpg.destroy_context()