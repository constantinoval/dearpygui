from libs.lsmesh_lib import lsdyna_model, node, element
from enum import IntEnum

class ShellType:
    plane_stress = 12
    plane_strain = 13
    axisymmetric_area = 14
    axisymmetric_volume = 15

class BaseLSmodel(lsdyna_model):
    def __init__(self, etype: ShellType=ShellType.axisymmetric_volume):
        super().__init__()
        self.etype = etype
        self.kwds['*CONTROL_HOURGLASS'] = ['         1\n']
        self.kwds['*CONTROL_ENERGY'] = ['         2         2         2         2\n']
        self.kwds['*CONTROL_SHELL'] = [(
            ' 20.000000         0        -1         0         2         2         1         0\n'
            '  1.000000         0         0         1         0\n'
            '         0         0         0         0         2\n'
            '         0         0         0         0         0         0  1.000000\n'
        )]
        self.kwds['*CONTROL_TIMESTEP'] = [(
            '     0.000  0.600000         0     0.000     0.000         0         0         0\n'
            '\n'
        )]
        self.kwds['*DATABASE_EXTENT_BINARY'] = [(
            '         6         6         3         1         1         1         1         1\n'
            '         0         1         0         1         1         1         2         1\n'
            '         0         0  1.000000         0         0         0STRESS    STRESS\n'
            '         0         0\n'
        )]
        self.kwds['*PART'] = [(
            'specimen\n'
            '         1         1         1         0         0         0         0         0\n'
        )]
        self.kwds['*SECTION_SHELL'] = [(
            '         1 {:9d}  1.000000         2  1.000000     0.000         0         1\n'
            '     0.000     0.000     0.000     0.000     0.000     0.000     0.000         0\n'
        ).format(etype)]
        self.kwds['*BOUNDARY_SPC_SET'] = [(
            '         1         0         1         1         0         0         0         0\n'
            '         2         0         1         0         0         0         0         0\n'
        )]
        self.kwds['*BOUNDARY_PRESCRIBED_MOTION_SET'] = [
            '         2         2         0        99       1.0         01.00000E28       0.0\n'
         ]

    def create_mesh(self, width: float, height: float, nx: int, ny: int):
        if width <= 0:
            raise ValueError('Ширина рабочей части должна быть неотрицательным числом')
        if height <= 0:
            raise ValueError('Высота рабочей части должна быть неотрицательным числом')
        if nx < 1:
            raise ValueError('Число разбиение по X должно быть не меньше 1')
        if ny < 1:
            raise ValueError('Число разбиение по Y должно быть не меньше 1')
        dx = width/nx
        dy = height/ny
        n = 1
        for j in range(0, ny+1):
            for i in range(0, nx+1):
                self.nodes[n] = node(n=n, crds=[i*dx, j*dy, 0])
                n += 1
        n = 1
        self.shells[1] = {}
        for j in range(0, ny):
            for i in range(0, nx):
                self.shells[1][n] = element(
                    n=n,
                    nodes=[j*(nx+1)+i+1, j*(nx+1)+i+2, (j+1)*(nx+1)+i+2, (j+1)*(nx+1)+i+1],
                    etype='*ELEMENT_SHELL',
                )
                n += 1

        self.nodesets[1] = list(range(1, nx+2))
        self.nodesets[2] = list(range((nx+1)*ny+1, (nx+1)*(ny+1)+1))
        self.nodesets_to_kwds()


if __name__=='__main__':
    m = BaseLSmodel(etype=ShellType.plane_stress)
    m.create_mesh(2.5, 10, 10, 80)
    m.save_model('test.k')
