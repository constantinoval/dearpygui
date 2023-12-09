import os
from libs.lsmesh_lib import lsdyna_model, boundBox
from libs.d3plot_reader.d3plot import D3plot
from libs.d3plot_reader.filter_type import FilterType
from libs.d3plot_reader.array_type import ArrayType
import numpy as np

class TrueDiagrammCalculator:
    def __init__(self) -> None:
        # путь к k-файлу входной модели
        self.model_path: str = ""
        # путь к решателю ls-dyna
        self.solver_path: str = ""
        # путь к рабочей директории проекта
        self._working_dir: str = os.path.join(os.path.abspath(os.curdir), "solution")
        # входная модель ls-dyna
        self.model: lsdyna_model | None = None
        # размер модели
        self.model_bbox: boundBox | None = None
        # массив с экспериментальными кривыми [время: list[float], скорость: list[float], сила: list[float]]
        self._exp_curves: list = []
        # число точек на расчитываемой кривой деформирования (оно же число записей при расчете)
        self._n_points: int = 10
        # приближение кривой деформирования в начале итерации:
        # [пластическая деформация: list[float], напряжение: list[float]]
        self.diag_0: list = []
        # приближение кривой деформирования в конце итерации
        self.diag_1: list = []
        # массив времен для записи результата
        self.dump_t: list[float] = []
        # массив приращений времени для записи результата
        self.dump_dt: list[float] = []
        # номер итерации
        self._iteration = 0
        #
        self.model_type = '2d'
        # результаты решения: [макс.пласт.деф: list[float], силы реакции: list[float]]
        self.solution_results: list = []
        # self.max_force_deflection = 1e6

    @property
    def iteration(self):
        "возвращает номер текущей итерации"
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        """
        присваивает номер текущей итерации
        если 0, то кривая деформирования сбрасывается на умолчание
        """
        self._iteration = value
        if value==0:
            self.assign_initial_diag()
            self.solution_results = []

    @property
    def working_dir(self):
        "возвращает рабочую директорию"
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value: str):
        "присваивает рабочую директорию. При необходимости создает папку"
        if not os.path.exists(value):
            os.makedirs(value)
        self._working_dir = value

    @property
    def exp_curves(self):
        "возвращает массив экспериментальных кривых"
        return self._exp_curves

    @exp_curves.setter
    def exp_curves(self, data: list):
        "присваивает массив экспериментальных кривых. расчитывает параметры записи результата при моделировании"
        self._exp_curves = data
        dt = data[0][-1] / (self.n_points-1)
        self.dump_t = [i*dt for i in range(self.n_points)]
        self.dump_dt = [dt]*(self.n_points)

    @property
    def n_points(self):
        return self._n_points

    @n_points.setter
    def n_points(self, value):
        if value!=self._n_points:
            self._n_points = value
            dt = self._exp_curves[0][-1] / (self.n_points-1)
            self.dump_t = [i*dt for i in range(self.n_points)]
            self.dump_dt = [dt]*(self.n_points)


    def assign_material_props(self, rho: float, E: float, nu: float, s0: float, Et: float) -> None:
        """
        Присваивает параметры материала
        Parameters
        ----------
        rho - плотность
        E   - модуль Юнга
        nu  - коэффициент Пуассона
        s0  - предел текучести
        Et  - касательный модуль пластического участка кривой деформирования

        -------

        """
        self.rho = rho
        self.E = E
        self.nu = nu
        self.s0 = s0
        self.Et = Et
        self.assign_initial_diag()

    def assign_initial_diag(self):
        "задает начальное приближение кривой деформирования"
        self.diag_0 = [[0, 100], [self.s0, self.s0+100*self.Et]]

    def assign_model(self, model_path: str) -> None:
        "ассоцияция модели. чтение карт. чтение сетки."
        if os.path.exists(model_path):
            print(f'Чтение сеточной модели из файла {model_path}...')
            self.model_path = model_path
            self.model = lsdyna_model(model_path, procIncludes=True)
            self.model_bbox = self.model.bbox
            self.model.proceed_node_sets()
        else:
            print(f'Файл {model_path} не найден')

    def read_exp_curves(self, file_path: str) -> None:
        """
        Чтение экспериментальных кривых из csv-файла.
        Parameters
        ----------
        file_path - путь к csv-файлу
        -------
        """

        if not os.path.exists(file_path):
            return
        t = []
        v = []
        f = []
        with open(file_path, 'r') as in_file:
            in_file.readline()
            for line in in_file:
                if not line.strip():
                    continue
                tt, vv, ff = [float(s) for s in line.split(',')[:3]]
                t.append(tt)
                v.append(vv)
                f.append(ff)
        self.exp_curves = [t, v, f]


    def save_base_model(self):
        """
        Сохранение базовой модели в рабочую папку
        -------

        """
        if not self.working_dir:
            print('Не задана рабочая директория')
            return
        savepath = os.path.join(self.working_dir, 'base.k')
        if self.model:
            with open(savepath, 'w') as fout:
                fout.write("*KEYWORD\n")
                for k, v in self.model.kwds.items():
                    if k.upper() == '*TITLE':
                        continue
                    for data in v:
                        fout.write(k+'\n')
                        fout.write(data)
                fout.write("*INCLUDE\nvar_part.k\nconst_part.k\nmesh.k\n*END\n")
            with open(os.path.join(self.working_dir, 'mesh.k'), 'w') as fout:
                fout.write('*KEYWORD\n')
                self.model.save_nodes(fout)
                self.model.save_elements(fout)
                fout.write('*END\n')

    def save_constant_part(self):
        """
        Сохранение неизменной на итерациях части расчетной модели:
        граничные условия (кривая скорости), окончание счета, запись узловых сил, запись результатов, карта модели материала
        -------

        """
        if not self.working_dir:
            print('Не задана рабочая директория')
            return
        savepart = os.path.join(self.working_dir, 'const_part.k')
        with open(savepart, 'w') as fout:
            fout.write('*KEYWORD\n')
            fout.write("""*DEFINE_CURVE
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
        99         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1
""")
            for t, v in zip(*(self.exp_curves[:2])):
                fout.write(f" {t:19g} {v:19g}\n")
            fout.write(f"""*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol
 {self.exp_curves[0][-1]:9g}         0       0.0       0.01.000000E8         0
""")
            fout.write("""*DATABASE_NODFOR
$#      dt    binary      lcur     ioopt
       0.0         0       100         1
*DATABASE_NODAL_FORCE_GROUP
$#    nsid       cid
         1         0
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
       0.0       100         0         0         0
$#   ioopt      rate    cutoff    window      type      pset
         0       0.0       0.0       0.0         0         0
""")
            fout.write(f"""*MAT_PIECEWISE_LINEAR_PLASTICITY
$#     mid        ro         e        pr      sigy      etan      fail      tdel
         1 {self.rho:9g} {self.E:9g} {self.nu:9g}       0.0       0.0       0.0       0.0
$#       c         p      lcss      lcsr        vp
       0.0       0.0       101         0       0.0
$#    eps1      eps2      eps3      eps4      eps5      eps6      eps7      eps8
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#     es1       es2       es3       es4       es5       es6       es7       es8
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
""")
            fout.write('*END\n')

    def save_variable_part(self):
        """
        Сохранение переменной части расчетной модели:
        кривая зависимости записи результатов, кривая зависимости напряжения от пластической деформации.
        -------
        """
        if not self.working_dir:
            print('Не задана рабочая директория')
            return
        savepart = os.path.join(self.working_dir, 'var_part.k')
        with open(savepart, 'w') as fout:
            fout.write('*KEYWORD\n')
            fout.write("""*DEFINE_CURVE
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
       100         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1
""")
            for t, dt in zip(self.dump_t, self.dump_dt):
                fout.write(f" {t:19g} {dt:19g}\n")
            fout.write("""*DEFINE_CURVE
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
       101         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1
""")
            for t, dt in zip(*self.diag_0):
                fout.write(f" {t:19g} {dt:19g}\n")
            fout.write(f" {10:19g} {self.diag_0[1][-1]:19g}\n")

            fout.write("*END\n")

    def parse_nodfor(self) -> list[list[float], list[float]]:
        """
        Извлечение зависимости силы реакции для узлового набора 1 из файла nodfor
        """
        if not os.path.exists(
            nodfor_path := os.path.join(self.working_dir, 'nodfor')
        ):
            print(f'Не найдено решение {nodfor_path}')
            return
        times = []
        forces = []
        with open(nodfor_path, 'r') as fin:
            for line in fin:
                if line.startswith(" n o d a l   f o r c e   g r o u p    o u t p u t"):
                    times.append(float(line.split()[-1]))
                    continue
                if line == " nodal group output number  1\n":
                    while not line.startswith("              xtotal="):
                        line = next(fin)
                    forces.append(float(line.strip().split()[3]))
        return times, forces


    def proc_solution_results(self):
        """
        Обработка результатов расчета последней итерации
        """
        if not os.path.exists(
            sol_path := os.path.join(self.working_dir, 'd3plot')
        ):
            print(f'Не найдено решение {sol_path}')
            return
        print(f'Анализ решения {sol_path}')
        d3plot = D3plot(
            sol_path,
            state_array_filter=[
                ArrayType.element_shell_effective_plastic_strain,
                'timesteps',
            ]
        )
        mask = d3plot.get_part_filter(filter_type=FilterType.SHELL, part_ids=[1])
        pstrain = d3plot.arrays[ArrayType.element_shell_effective_plastic_strain][:, mask]
        times1 = d3plot.arrays['timesteps']
        maxep = np.max(np.max(pstrain, axis=1), axis=1)
        times2, forces = self.parse_nodfor()
        self.solution_results = [
            np.interp(self.dump_t, times1, maxep),
            np.interp(self.dump_t, times2, forces),
        ]

    @property
    def max_force_error(self):
        if not self.solution_results:
            return
        f_exp = np.interp(self.dump_t, self.exp_curves[0], self.exp_curves[2])
        return np.abs((f_exp[1:]-self.solution_results[1][1:])/f_exp[1:]).max()*100

    def correct_material_diagramm(self):
        st = np.interp(self.solution_results[0], self.diag_0[0], self.diag_0[1])
        f_exp = np.interp(self.dump_t, self.exp_curves[0], self.exp_curves[2])
        self.diag_0 = [[0], [1]]
        for i in range(1, self.n_points):
            self.diag_0[0].append(self.solution_results[0][i])
            # if (np.abs((f_exp[i]-self.solution_results[1][i])/f_exp[i]))>0.05:
            self.diag_0[1].append(st[i]*f_exp[i]/self.solution_results[1][i])
            # else:
            #     self.diag_0[1].append(st[i])
        # e = np.linspace(0, self.solution_results[0][-1], self.n_points)
        # self.dump_t = np.interp(e, self.solution_results[0], self.dump_t)
        # self.dump_dt = np.diff(self.dump_t, 1)

    def run_calculation(self, ncpus=1):
        if not os.path.exists(task_path := os.path.join(self.working_dir, 'base.k')):
            print(f'Не обнаружена задача {task_path}')
            return
        with open(bat_path := os.path.join(self.working_dir, 'run.bat'), 'w') as bat_file:
            bat_file.write(f"cd /d {self.working_dir}\n")
            bat_file.write(f"{self.solver_path} i=base.k ncpus={ncpus}\n")
        try:
            os.system(bat_path)
        except Exception:
            print('Не удалось посчитать. Проверьте папку проекта.')

    @property
    def strain(self):
        if not self.diag_0:
            return []
        ee = self.diag_0[0][0]/self.E
        return [0] + [ep+ee for ep in self.diag_0[0]]

    @property
    def stress(self):
        if not self.diag_0:
            return []
        return [0] + list(self.diag_0[1])

if __name__=='__main__':
    calc = TrueDiagrammCalculator()
    calc.assign_model('../model/main.k')
    calc.working_dir = "d:/111/"
    calc.solver_path = 'run_dyna'
    calc.n_points = 10
    calc.read_exp_curves('../model/exp_data.csv')
    calc.assign_material_props(7.8e-3, 200000, 0.28, 200, 1000)
    calc.save_base_model()
    calc.save_constant_part()
    calc.save_variable_part()
    calc.run_calculation()
    calc.proc_solution_results()
    print(calc.solution_results)