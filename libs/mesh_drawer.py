import dearpygui.dearpygui as dpg
from libs.lsmesh_lib import lsdyna_model


class MeshDrawer:
    def __init__(self, width, height, bg_color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        with dpg.stage() as self.id:
            with dpg.drawlist(width=self.width, height=self.height):
                with dpg.draw_layer() as self.bg:
                    dpg.draw_rectangle(
                        pmin=(0, 0),
                        pmax=(self.width, self.height),
                        fill=self.bg_color
                    )
                with dpg.draw_layer() as self.draw_layer:
                    pass
        self._model: None | lsdyna_model = None
        self.part_colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        )
        
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value: lsdyna_model):
        self._model = value
        self._bbox = self._model.bbox
        self.scale_factor = 0.9 * min(self.width / self._bbox.a,
                                      self.height / self._bbox.b)
        self.x_draw = lambda x: 0.05 * self.width + self.scale_factor * (x - self._bbox.xmin)
        self.y_draw = lambda y: 0.95 * self.height - self.scale_factor * (y - self._bbox.ymin)
        self.draw_mesh()
        self.draw_bcs()
    
    def sumbit(self, parent):
        dpg.push_container_stack(parent)
        dpg.unstage(self.id)
        dpg.pop_container_stack()

    def draw_mesh(self):
        dpg.delete_item(self.draw_layer, children_only=True)
        for i, part in enumerate(self.model.shells):
            for sh in self.model.shells[part].values():
                pnts = [
                    (
                        self.x_draw(self.model.nodes[n].x),
                        self.y_draw(self.model.nodes[n].y),
                     ) for n in sh.nodes
                ]
                pnts.append(pnts[0])
                dpg.draw_polygon(pnts, fill=self.part_colors[i], color=(0, 0, 0), thickness=1, parent=self.draw_layer)
    
    def draw_bcs(self):
        if self.model:
            self.model.proceed_node_sets()
        if 1 in self.model.nodesets:
            for n in self.model.nodesets[1]:
                dpg.draw_circle(
                    center=(
                        self.x_draw(self.model.nodes[n].x),
                        self.y_draw(self.model.nodes[n].y),
                    ),
                    radius=3,
                    fill=(0, 255, 0),
                    color=(0, 0, 0),
                    thickness=1,
                    parent=self.draw_layer
                )
        if 2 in self.model.nodesets:
            for n in self.model.nodesets[2]:
                x = self.x_draw(self.model.nodes[n].x)
                y = self.y_draw(self.model.nodes[n].y)
                dpg.draw_arrow(
                    p1=(x, y - 10),
                    p2=(x, y),
                    color=(0, 0, 255),
                    thickness=1,
                    parent=self.draw_layer
                )
    
if __name__=="__main__":
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    with dpg.window(label="Example Window", width=800, height=600, tag='main'):
        with dpg.child_window() as w:
            drawer = MeshDrawer(width=500, height=500)
            drawer.sumbit(parent=w)
            mesh = lsdyna_model('../model/indent_sph_ak4/sph.k')
            drawer.model = mesh
    dpg.show_viewport()
    dpg.set_primary_window('main', True)
    dpg.start_dearpygui()
    dpg.destroy_context()