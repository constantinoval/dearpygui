from math import ceil
import os
from pathlib import Path
import tempfile as tfile
import subprocess


class LSRunner:
    def __init__(self, solver_string: str = 'run_dyna', progress_width: int = 50,
                 ncpus=4):
        self.solver_string = solver_string
        self.progress_width = progress_width
        self.ncpus = ncpus
        
    def run_task(self, task_file: str):
        print(f'Запуск задачи {task_file} на {self.ncpus} CPU')
        p = Path(task_file)
        task_dir = p.parent
        file_name = p.name
        temp_file=tfile.NamedTemporaryFile(mode="w", suffix=".bat", prefix="ls_run_",
                                           dir=os.curdir, delete=False)
        temp_file.write(f'cd /d {task_dir}\n')
        temp_file.write(f'run_dyna i={file_name} ncpus=-{self.ncpus}\n')
        temp_file.close()
        p = subprocess.Popen(temp_file.name, stdout=subprocess.PIPE)
        termination_time = 0
        normal_termination = False
        solution_time = ''
        for line in iter(p.stdout.readline, b''):
            line = line.decode()[:-2]
            if line.startswith(' termination time'):
                termination_time = float(line.split()[-1])
            if ('write d3plot file' in line) or ('flush i/o buffers' in line):
                if termination_time == 0:
                    continue
                current_time = float(line.split()[2])
                progress = ceil(current_time/termination_time*self.progress_width)
                print('\r['+'#'*progress + '-'*(self.progress_width-progress)+
                      f'] {progress/self.progress_width*100: 5.0f}%' + f" cur_time={current_time: 7.3g}, tot_time={termination_time: 7.3g}", end='')
                continue
            if 'N o r m a l    t e r m i n a t i o n' in line:
                normal_termination = True
                continue
            if 'Total CPU time     =' in line:
                solution_time = line.split('(')[-1][:-1]
        p.wait()
        rez = 'успешно' if normal_termination else 'ошибка'
        print(f' {rez}')
        if solution_time:
            print(f'Время счета: {solution_time}')
        p.stdout.close()
        os.remove(temp_file.name)
        
if __name__ == '__main__':
    lr = LSRunner()
    lr.run_task('d:/111/base.k')
    