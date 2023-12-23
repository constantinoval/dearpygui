import dearpygui.dearpygui as dpg


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
        self._model = None
        
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._bbox = self._model.bbox
        self.scale_factor = 0.9 * min(self.width / self._bbox.a,
                                      self.height / self._bbox.b)
        self.x_draw = lambda x: 0.05 * self.width + self.scale_factor * (x - self._bbox.xmin)
        self.y_draw = lambda y: 0.95 * self.height - self.scale_factor * (y - self._bbox.ymin)
    
    def sumbit(self, parent):
        dpg.push_container_stack(parent)
        dpg.unstage(self.id)
        dpg.pop_container_stack()

    def draw_mesh(self):
        pass
    #     x_draw = lambda x: 0.05 * MESH_DRAW_AREA_WIDTH + k * (x - calculator.model_bbox.xmin)
    #     y_draw = lambda y: 0.95 * MESH_DRAW_AREA_HEIGH - k * (y - calculator.model_bbox.ymin)
    #     dpg.delete_item(dpg_draw_layer_tag, children_only=True)
    #     for part in calculator.model.shells:
    #         for sh in calculator.model.shells[part].values():
    #             pnts = [
    #                 (x_draw(calculator.model.nodes[n].x), y_draw(calculator.model.nodes[n].y)) for n in sh.nodes
    #             ]
    #             pnts.append(pnts[0])
    #             dpg.draw_polygon(pnts, fill=(255, 0, 0), color=(0, 0, 0), thickness=1, parent=dpg_draw_layer_tag)
    
    def draw_bcs(self):
        pass
    #     if 1 in calculator.model.nodesets:
    #         for n in calculator.model.nodesets[1]:
    #             dpg.draw_circle(
    #                 center=(
    #                     x_draw(calculator.model.nodes[n].x),
    #                     y_draw(calculator.model.nodes[n].y),
    #                 ),
    #                 radius=3,
    #                 fill=(0, 255, 0),
    #                 color=(0, 0, 0),
    #                 thickness=1,
    #                 parent=dpg_draw_layer_tag
    #             )
    #     if 2 in calculator.model.nodesets:
    #         for n in calculator.model.nodesets[2]:
    #             x = x_draw(calculator.model.nodes[n].x)
    #             y = y_draw(calculator.model.nodes[n].y)
    #             dpg.draw_arrow(
    #                 p1=(x, y - 10),
    #                 p2=(x, y),
    #                 color=(0, 0, 255),
    #                 thickness=1,
    #                 parent=dpg_draw_layer_tag
    #             )
    
