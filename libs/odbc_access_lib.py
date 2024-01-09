# -- coding: utf8
import numpy as np
import pyodbc
import re
# matplotlib.use('PS')
# %%

number = re.compile(r'[+-]?[0-9]+([\.]?[0-9]+)?([eE]?[+-]?[0-9]+)?')


def runningMeanFast(x, N):
    rez = np.roll(np.convolve(x, np.ones((N,))/N)[(N-1):], N//2)
    rez[:N//2] = rez[N//2]
    return rez


def toReal(diag, exp_type):
    sign = [-1, 1][exp_type == 't']
    diag['der'] = np.array(diag['det'])/(1+sign*np.array(diag['et']))
    diag['sr'] = np.array(diag['st'])*(1+sign*np.array(diag['et']))
    diag['er'] = sign*np.log(1+sign*np.array(diag['et']))


def meanDE(data):
    maxe = data['et'].max()
    N1 = (data['det']*(1-data['et']/maxe)).argmax()
    N2 = (data['st']*(1+data['et']/maxe)).argmax()
#    plt.plot(data[3]*(1-data[1]/maxe))
#    plt.show()
    if N2 < N1:
        N1, N2 = N2, N1
    N = N2-N1
    return data['det'][N1+N//5:N2-N//5].mean()  # , N1+N//3, N2-N//3


def integrate(y, dx):
    rez = [0]
    for i in range(1, len(y)):
        rez.append(rez[-1]+0.5*(y[i]+y[i-1])*dx)
    return np.array(rez)


def calcDiagram(dt, ein, eref, etr, cfg):
    n = len(ein)
    et = []
    st = []
    det = []
    for i in range(n):
        V1 = cfg['c1']*(ein[i]+eref[i])
        V2 = cfg['c2']*etr[i]
        det.append((V1-V2)/cfg['Lsp'])
        F = cfg['E2']*cfg['S2']*etr[i]
        st.append(F/cfg['Ssp'])
        et.append(integrate(det[i], dt))
    for i in range(1, n):
        et[i] += et[i-1][-1]
    et = np.array(et).flatten()
    det = np.array(det).flatten()
    st = np.array(st).flatten()
    return et, st, det


def calcDiagram2(db, exp_code):
    experiment = db.getExperimentData(exp_code)
    b1 = db.getBarData(experiment.bars[0])
    b2 = db.getBarData(experiment.bars[1])
    t = experiment.pulses['t']
    return t, calcDiagram(dt=(t[1]-t[0]),
                          ein=[experiment.pulses['pulses'][0]],
                          eref=[experiment.pulses['pulses'][1]],
                          etr=[experiment.pulses['pulses'][2]],
                          cfg={'E2': b2.E, 'c1': b1.c, 'c2': b2.c, 'S2': b2.S,
                               'Ssp': experiment.d0**2/4.*3.14, 'Lsp': experiment.l0*1e-3}
                          )


def tofloat(s):
    if s == None:
        return 0
    if type(s) != str:
        return float(s)
    i = 0
    for c in s:
        if not c.isdigit() and c != '.':
            break
        i += 1
    if not i:
        return 0.0
    return float(s[:i])


def unpackTable(tbl):
    if not tbl:
        return [], []
    # tmp = tbl.split()
    try:
        tmp = [num.group(0) for num in number.finditer(tbl)]
        N = int(tmp[0])
        dt = float(tmp[1])
        t = np.arange(N)*dt
        NN = (len(tmp)-2)//N
        cols = []
        for i in range(NN):
            cols.append(np.array(list(map(float, tmp[2+N*i:2+N*(i+1)]))))
        return t, cols
    except Exception as e:
        print(e)
        return [], []


def packTable(t, cols):
    rez = '{0:d}\t\n{1}\t\n'.format(len(t), t[1]-t[0])
    for c in cols:
        for num in c:
            rez += '{}\t'.format(num)
        rez += '\t\n'
    return rez


class odbc:
    def __init__(self, dbFile):
        self.conn = pyodbc.connect(
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; DBQ=%s;" % dbFile)
        self.cursor = self.conn.cursor()

    def getInfo(self, table, fieldsCond='', fieldsCondValues='', getFields='*', cond=' and '):
        if type(getFields) not in [list, tuple]:
            getFields = [getFields]
        getFields = list(getFields)
        for i in range(len(getFields)):
            if getFields[i] != '*':
                getFields[i] = '['+getFields[i]+']'
        getFields = ' , '.join(getFields)
        if fieldsCond and fieldsCondValues:
            if type(fieldsCond) == str:
                fieldsCond = [fieldsCond]
            if type(fieldsCondValues) not in [list, tuple]:
                fieldsCondValues = [fieldsCondValues]
            if len(fieldsCond) != len(fieldsCondValues):
                return []
            fieldsCond = list(fieldsCond)
            N = len(fieldsCond)
            for i in range(N):
                fieldsCond[i] = '['+fieldsCond[i]+']'
            fieldsCond = list(fieldsCond)
            for i, f in enumerate(fieldsCond):
                fieldsCond[i] = f+'=?'
            fieldsCond = cond.join(fieldsCond)
        s = 'select {0} from {1} '.format(getFields, table)
        if fieldsCond and fieldsCondValues:
            s += 'where {0}'.format(fieldsCond)
        if fieldsCondValues:
            data = self.cursor.execute(s, *fieldsCondValues).fetchall()
        else:
            data = self.cursor.execute(s).fetchall()
        allrez = []
        for d in data:
            rez = {}
            for i, c in enumerate(d.cursor_description):
                rez[c[0]] = d[i]
            allrez.append(rez)
        return allrez

    def putInfo(self, table, putFields='', putFieldsValues='', fieldsCond='', fieldsCondValues='', cond=' and '):
        if type(putFields) == str:
            putFields = [putFields]
        if type(putFieldsValues) not in [list, tuple]:
            putFieldsValues = [putFieldsValues]
        if len(putFields) != len(putFieldsValues):
            return 0
        putS = []
        for f in putFields:
            putS.append('[{}] = ?'.format(f))
        putS = ' , '.join(putS)
        if fieldsCond and fieldsCondValues:
            if type(fieldsCond) == str:
                fieldsCond = [fieldsCond]
            if type(fieldsCondValues) not in [list, tuple]:
                fieldsCondValues = [fieldsCondValues]
            if len(fieldsCond) != len(fieldsCondValues):
                return 0
            fieldsCond = list(fieldsCond)
            N = len(fieldsCond)
            for i in range(N):
                fieldsCond[i] = '['+fieldsCond[i]+']'
            for i, f in enumerate(fieldsCond):
                fieldsCond[i] = f+'=?'
            fieldsCond = cond.join(fieldsCond)
        s = 'update {0} set {1} '.format(table, putS)
        if fieldsCond and fieldsCondValues:
            s += 'where {0}'.format(fieldsCond)
        if fieldsCondValues:
            vals = list(putFieldsValues)+list(fieldsCondValues)
            self.cursor.execute(s, *vals)
        else:
            self.cursor.execute(s, *putFieldsValues)
        self.cursor.commit()
        return 1

    def insertInfo(self, table, putFields='', putFieldsValues='', commit=True):
        if type(putFields) == str:
            putFields = [putFields]
        for i in range(len(putFields)):
            putFields[i] = '['+putFields[i]+']'
        if type(putFieldsValues) not in [list, tuple]:
            putFieldsValues = [putFieldsValues]
        if len(putFields) != len(putFieldsValues):
            return 0
        putS = ' , '.join(putFields)
        s = 'insert into {0}({1}) values ({2})'.format(
            table, putS, ' , '.join(['?']*len(putFields)))
        self.cursor.execute(s, *putFieldsValues)
        if commit:
            self.cursor.commit()
        return 1

    def close(self):
        self.conn.close()


class bar(object):
    def __init__(self, odbcRez):
        self.E = tofloat(odbcRez['МодульУпругости(МПа)'])
        self.code = str(odbcRez['КодМернСтерж'])
        self.mat = str(odbcRez['Материал'])
        self.d = tofloat(odbcRez['Диаметр(мм)'])
        self.d0 = tofloat(odbcRez['ВнутреннийДиаметр'])
        self.c = tofloat(odbcRez['СкоростьЗвука(мсек)'])
        self.l = tofloat(odbcRez['Длина(мм)'])
        self.S = 0.25*np.pi*(self.d**2-self.d0**2)
        self.dispersion_data = odbcRez['Дисперсия']

    def __repr__(self):
        rez = ''
        rez += f'Код стержня: {self.code}\n'
        rez += f'Материал стержня: {self.mat}\n'
        rez += f'E = {self.E} МПа\n'
        rez += f'c = {self.c} м/c\n'
        rez += f'l = {self.l} мм\n'
        rez += f'd = {self.d} мм\n'
        rez += f'd0 = {self.d0} мм\n'
        return rez


class striker(object):
    def __init__(self, odbcRez):
        self.code = str(odbcRez['КодУдарника'])
        self.mat = str(odbcRez['МатериалУдарника'])
        self.d = tofloat(odbcRez['ДиаметрУдарника(мм)'])
        self.l = tofloat(odbcRez['ДлинаУдарника(мм)'])
        self.S = np.pi*self.d**2/4.

    def __repr__(self):
        rez = ''
        rez += f'Код ударника: {self.code}\n'
        rez += f'Материал ударника: {self.mat}\n'
        rez += f'l = {self.l} мм\n'
        rez += f'd = {self.d} мм\n'
        return rez


class experimentalData(object):
    def __init__(self, odbcRez):
        self.data = str(odbcRez['Дата']).split()[0]
        self.code = str(odbcRez['КодОбразца'])
        self.striker = str(odbcRez['Ударник'])
        self.expType = str(odbcRez['ТипЭксперимента'])
        self.T = tofloat(odbcRez['Температура'])
        self.P = tofloat(odbcRez['ДавлениеКВД'])
        self.V = tofloat(odbcRez['СкоростьУдарника'])
        self.d0 = tofloat(odbcRez['Диаметр'])
        self.l0 = tofloat(odbcRez['Длина'])
        self.l = tofloat(odbcRez['ОстаточнаяДлина'])
        self.d = tofloat(odbcRez['Шейка'])
        self.note = str(odbcRez['Примечание'])
        self.osc = {}
        self.osc['t'], self.osc['rays'] = unpackTable(odbcRez['Осциллограмма'])
        self.pulses = {}
        self.pulses['t'], self.pulses['pulses'] = unpackTable(
            odbcRez['ИмпульсыОбработанные'])
        self.tarir = [
            tofloat(odbcRez['КалибровочныйКоэффициентНС']),
            tofloat(odbcRez['КалибровочныйКоэффициентОС']),
            tofloat(odbcRez['КалибровочныйКоэффициентОС2(Обоймы)'])
        ]
        self.datPosition = [
            tofloat(odbcRez['ПоложениеДатчиковНС(мм)']),
            tofloat(odbcRez['ПоложениеДатчиковОС(мм)'])
        ]
        self.bars = [
            str(odbcRez['НагружающийСтержень']),
            str(odbcRez['ОпорныйСтержень']),
            str(odbcRez['ОпорныйСтержень2(Обойма)'])
        ]

    def __repr__(self):
        rez = ''
        rez += f'Код эксперимента: {self.code}\n'
        return rez


class expODBC(odbc):
    def __init__(self, dbFile):
        super().__init__(dbFile)

    def getExpTypes(self):
        return self.getInfo(getFields=('ТипЭксперимента', 'КодЭксперимента'), table='ТипЭксперимента')

    def getMaterials(self):
        return self.getInfo(getFields=('Материал', 'КодМатериала'), table='МатериалЭксперимент')

    def getNumbers(self, expType, materialCode):
        materialCode = tofloat(materialCode)
        return self.getInfo(getFields='НомерОбразца', table='Эксперимент',
                            fieldsCond=('ТипЭксперимента', 'КодМатериала'),
                            fieldsCondValues=(expType, materialCode))

    def getExperimentData(self, sampleCode):
        rez = self.getInfo(table='Эксперимент', fieldsCond='КодОбразца',
                           fieldsCondValues=sampleCode)
        if rez:
            return experimentalData(rez[0])
        else:
            print(f'В базе не найден эксперимент с кодом {sampleCode}')
            return None

    def getStrickerData(self, strickerCode):
        rez = self.getInfo(table='Ударник', fieldsCond='КодУдарника',
                           fieldsCondValues=strickerCode)
        if rez:
            return striker(rez[0])
        else:
            print(f'В базе не найден ударник с кодом {strickerCode}')
            return None

    def getBarData(self, barCode):
        rez = self.getInfo(table='МерныйСтержень', fieldsCond='КодМернСтерж',
                           fieldsCondValues=barCode)
        if rez:
            return bar(rez[0])
        else:
            print(f'В базе не найден стержень с кодом {barCode}')
            return None

    def putOsc(self, sampleCode, data):
        self.putInfo(table='Эксперимент', putFields='Осциллограмма', putFieldsValues=data,
                     fieldsCond='КодОбразца', fieldsCondValues=sampleCode)

    def putPulses(self, sampleCode, data):
        self.putInfo(table='Эксперимент', putFields='ИмпульсыОбработанные', putFieldsValues=data,
                     fieldsCond='КодОбразца', fieldsCondValues=sampleCode)

    def getDiagram(self, exp_code):
        exp_data = self.getExperimentData(exp_code)
        if not len(exp_data.pulses['t']):
            return None
        if exp_code[:2] == 'di':
            return self.calcDiagram_DI(exp_code)
        et = []
        st = []
        det = []
        b1 = self.getBarData(exp_data.bars[0])
        b2 = self.getBarData(exp_data.bars[1])
        Ssp = exp_data.d0**2/4.*np.pi
        V = b1.c*(exp_data.pulses['pulses'][0]+exp_data.pulses['pulses']
                  [1])-b2.c*exp_data.pulses['pulses'][2]
        U = integrate(V, exp_data.pulses['t'][1]-exp_data.pulses['t'][0])
        det = V/exp_data.l0*1000
        F = b2.S*b2.E*exp_data.pulses['pulses'][2]
        st = F/Ssp
        et = integrate(det, exp_data.pulses['t'][1]-exp_data.pulses['t'][0])
        return {'t': exp_data.pulses['t'], 'det': det, 'st': st,
                'et': et, 'v': V, 'u': U, 'F': F}

    def calcDiagram_DI(self, exp_code: str):
        if exp_code[:2] != 'di':
            return
        experiment = self.getExperimentData(exp_code)
        b1 = self.getBarData(experiment.bars[0])
        t = experiment.pulses['t']*1e6
        ei = experiment.pulses['pulses'][0]
        striker = self.getStrickerData(experiment.striker)
        Ls = striker.l*1e-3
        cs = b1.c
        c = cs
        ttt = np.linspace(0, t[-1], len(t))
        e_mod = np.interp(ttt, t, ei) + 2 * \
            np.interp(ttt, t+2*Ls/cs*1e6, ei, left=0)
        e_mod = np.interp(ttt, ttt, e_mod) + 2 * \
            np.interp(ttt, ttt+4*Ls/cs*1e6, e_mod, left=0)
        e = np.interp(ttt, t, ei)
        E = b1.E
        S = b1.S
        d0 = experiment.d0*1e-3
        l0 = experiment.l0*1e-3
        F = e*E*S
        st = F/np.pi/d0**2*4*1e-6
        V = experiment.V-c*e-cs*e_mod
        U = integrate(V, (t[1]-t[0])*1e-6)
        det = V/l0
        et = integrate(det, (t[1]-t[0])*1e-6)
        return {'t': ttt*1e-6, 'det': det, 'st': st,
                'et': et, 'v': V, 'u': U, 'F': F}


# def smooth_stress(exp, pow=2, width=5):
#     """exp - dict {'et', 'st', 'det', etc.}
# """
#     st = exp['st']
#     idxs = st > 0.5*st.max()
#     st = st[idxs]
#     p1 = find_peaks(st, width=width)[0]
#     p2 = find_peaks(-st, width=width)[0]
#     i1 = min(p1[0], p2[0])
#     i2 = max(p1[-1], p2[-1])
#     p = np.polyfit(range(i1, i2), st[i1:i2], pow)
#     x = range(i1, i2)
#     y = np.polyval(p, range(i1, i2))
#     st[x] = y
#     def pf(x): return np.polyval(p, x)
#     for i, sst in enumerate(st):
#         if sst > pf(i):
#             break
#     xx = [i-2, i-1, i1, i1+1]
#     yy = st[xx]
#     xxx = range(i - 2, i1 + 1)
#     f = interp1d(xx, yy, kind='cubic')
#     st[xxx] = f(xxx)
#     for i in range(i2, len(st)):
#         if st[i] < pf(i):
#             break
#     if i-i2 > 0:
#         print('ok')
#         xx = range(i2, i)
#         st[xx] = pf(xx)
#     exp['st'][idxs] = st



if __name__ == '__main__':
    pass
    # import matplotlib.pylab as plt
    # dbFile = r"f:\experiments\db\ЭкспериментальныеДанные2023_обработанные.accdb"
    # db = expODBC(dbFile)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # stress_level = 1500
    # strain_at_stress_level = 0
    # for exp in [
    #     'di707-05(1-8)',
    #     'di707-10(1-8)',
    #     'di707-11(1-8)',
    #     'di707-12(1-8)',
    #     'di707-18(2-8)',
    # ]:
    #     d = db.getDiagram(exp)
    #     # plt.plot(d[0], d[-1])
    #     # plt.plot(d[0], d[-2])
    #     N = np.searchsorted(d['st'], stress_level)
    #     if strain_at_stress_level == 0:
    #         strain_at_stress_level = d['et'][N]
    #         de = 0
    #     else:
    #         de = d['et'][N]-strain_at_stress_level
    #     l, = ax.plot(d['et']-de, d['st'], label=exp)
    #     ax2.plot(d['et']-de, d['det'], '--', color=l.get_color())
    # ax.legend()
    # ax.set_xlabel('деформация')
    # plt.show()


# %%
