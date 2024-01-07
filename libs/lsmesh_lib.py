# -*- coding: utf-8 -*-
from __future__ import print_function, generators, division, with_statement
import numpy as np
import os
from math import sqrt, sin, cos, pi, acos
from copy import deepcopy
from bisect import bisect_left
from os import system, remove, curdir, listdir
from collections import defaultdict


def shift(l, idx):
    return l[idx:] + l[:idx]


def splitByN(l, n):
    if n >= len(l):
        return [l]
    nn = len(l) // n
    rez = []
    for i in range(nn):
        rez.append(l[i * n:(i + 1) * n])
    if len(l) % n:
        rez.append(l[nn * n:])
    return rez


def splitByNseq(s, n):
    while sum(n) + n[-1] <= len(s):
        n.append(n[-1])
    pos = [sum(n[:i + 1]) for i in range(len(n))]
    pos.insert(0, 0)
    rez = []
    for i in range(len(pos) - 1):
        rez.append(s[pos[i]:pos[i + 1]].strip())
    return rez


def setInd(n, s, isSorted=False):
    ss = s
    if not isSorted:
        ss = deepcopy(s)
        ss.sort()
    idx = bisect_left(ss, n)
    if idx == len(ss):
        idx = -1
    if idx == 0 and ss[0] != n:
        idx = -1
    if s[idx] != n:
        idx = -1
    return idx


def isInSet(n, s, isSorted=False):
    return setInd(n, s, isSorted) != -1


def isSubSet(s1, s2, isSorted=False):
    ss = s2
    if not isSorted:
        ss = deepcopy(s2)
        ss.sort()
    rez = True
    for n in s1:
        rez = rez and (isInSet(n, ss, isSorted=True) != -1)
    return rez


def formatSetKeyword(data, num=1, kw='*SET_NODE_LIST'):
    n = len(data)
    if not n: return ""
    rez = '{:10d}\n'.format(num)
    fs = ("{:>10}" * 8 + "\n") * (n // 8)
    if n % 8:
        fs += "{:>10}" * (n % 8) + "\n"
    rez += fs.format(*data)
    return rez


class ConsoleProgress(object):
    def __init__(self, total, scaleLength=50, text='Progress=',
                 symbol='-#'):
        self.current = 0
        self.total = total
        self.chekRedraw = lambda i: not (i % (total // scaleLength))
        self.scaleLength = scaleLength
        self.text = text
        self.symbol = symbol

    def redraw(self):
        if self.chekRedraw(self.current):
            print(progress(self.current, self.total, self.scaleLength, self.text, self.symbol), end='')  # , flush=True)
        self.current += 1

    def finalize(self):
        print(progress(self.total, self.total, self.scaleLength, self.text, self.symbol))


def progress(cur, total, scaleLength=20, text='Progress=', symbol='-#'):
    if total != 0:
        prog = int(100. * cur / total)
    else:
        prog = 100
    rez = '\r{0}{1:4}% '.format(text, prog)
    numD = int(prog * scaleLength / 100.)
    rez += symbol[1] * numD + symbol[0] * (scaleLength - numD)
    rez += ' Total={0:10}, current={1:10}'.format(total, cur)
    return rez


class Point(object):
    def __init__(self, x, y, z):
        """
        Point(x,y,z) - create point(vector) in 3D
        """
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        """
        Length of the vector
        """
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        """
        Return vector of unity length
        """
        l = self.norm()
        if l == 0:
            return self
        p = Point(self.x / l, self.y / l, self.z / l)
        return p

    def __repr__(self):
        return '(%f,%f,%f)' % (self.x, self.y, self.z)

    def dot(self, p2):
        rez = self.x * p2.x + self.y * p2.y + self.z * p2.z
        return rez

    def cross(self, p2):
        x = self.y * p2.z - self.z * p2.y
        y = self.z * p2.x - self.x * p2.z
        z = self.x * p2.y - self.y * p2.x
        return Point(x, y, z)

    def scaled(self, a):
        return Point(self.x * a, self.y * a, self.z * a)

    def __add__(self, p2):
        return Point(self.x + p2.x, self.y + p2.y, self.z + p2.z)

    def __sub__(self, p2):
        return Point(self.x - p2.x, self.y - p2.y, self.z - p2.z)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    def shuff(self, p1, p2):
        return (self.cross(p1)).dot(p2)

    def _getData(self):
        return np.array([self.x, self.y, self.z])

    def _setData(self, coords):
        coords = list(coords) + [0] * (3 - len(coords))
        self.x, self.y, self.z = coords

    data = property(_getData, _setData)

    def angle(self, p2, direction_vector=None):
        p1_u = self.normalized()
        p2_u = p2.normalized()
        rez = np.arccos(
            np.clip(np.dot(p1_u.data, p2_u.data), -1.0, 1.0)) * 180. / np.pi
        if direction_vector:
            if self.cross(p2).dot(direction_vector) < 0:
                rez = 360 - rez
        return rez


class Quaternion(object):
    data = [0, 0, 0, 0]
    R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def __init__(self, angle, v):
        """
        Quaternion(angle, vector)
        Creates quanternion with angle - angle, abount axis - vector.
        """
        self.a = angle
        self.v = v.normalized()
        a2 = angle / 2.
        self.data[0] = cos(a2)
        sina2 = sin(a2)
        d = self.v.data
        for i in range(3):
            self.data[i + 1] = d[i] * sina2
        self.calculateRotationMatrix()

    def calculateRotationMatrix(self):
        w = self.data[0]
        x = self.data[1]
        y = self.data[2]
        z = self.data[3]
        self.R[0][0] = 1 - 2 * y * y - 2 * z * z
        self.R[0][1] = 2 * x * y - 2 * z * w
        self.R[0][2] = 2 * x * z + 2 * y * w
        self.R[1][0] = 2 * x * y + 2 * z * w
        self.R[1][1] = 1 - 2 * x * x - 2 * z * z
        self.R[1][2] = 2 * y * z - 2 * x * w
        self.R[2][0] = 2 * x * z - 2 * y * w
        self.R[2][1] = 2 * y * z + 2 * x * w
        self.R[2][2] = 1 - 2 * x * x - 2 * y * y

    def rotatePoint(self, p):
        rez = [0, 0, 0]
        d = p.data
        for i in range(3):
            rez[i] = 0
            for j in range(3):
                rez[i] += self.R[i][j] * d[j]
        return Point(rez[0], rez[1], rez[2])

    def _getData(self):
        return [self.data[i] for i in range(4)]

    quaterionData = property(_getData)

    def _getRotationMatrix(self):
        rez = []
        for i in range(3):
            rez.append([])
            for j in range(3):
                rez[i].append(self.R[i][j])
        return rez

    rotationMatrix = property(_getRotationMatrix)


def searchKeywords(data):
    comments = []
    for i, l in enumerate(data):
        if l.startswith('$'):
            comments.append(i)
    comments.reverse()
    for i in comments:
        data.pop(i)
    keywords = []
    indexes = []
    for i, l in enumerate(data):
        if l.startswith('*'):
            keywords.append(l.upper()[:-1])
            indexes.append(i)
    return keywords, indexes


def readData(fname, procedIncludes=False):
    data = open(fname, 'r').readlines()
    dirName = os.path.dirname(os.path.realpath(fname)) + os.path.sep
    keywords, indexes = searchKeywords(data)
    if procedIncludes and keywords.count('*INCLUDE'):
        incFiles = []
        for i, kw in enumerate(keywords):
            if kw == '*INCLUDE':
                incFiles.extend(data[indexes[i] + 1:indexes[i + 1]])
        for f in incFiles:
            print('Proced includes ', f[:-1])
            data.extend(readData(dirName + f[:-1], True))
    return data


def calcV4points(points):
    p = [np.array(pp) for pp in points]
    m = np.matrix([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
    return abs(np.linalg.det(m) / 6.)


def read_set(fname, stype='NODE', num=None):
    if isinstance(num, int):
        num = [num]
    rez = {}
    f = open(fname, 'r')
    l = next(f)
    while 1:
        if l.upper().startswith('*SET_' + stype.upper()):
            while 1:
                try:
                    nset = int(l[:10])
                    break
                except:
                    pass
                l = next(f)
            if num:
                if not nset in num:
                    l = next(f)
                    continue
            rez[nset] = []
            l = next(f)
            while not l.startswith('*'):
                try:
                    rez[nset] += [int(ll) for ll in l.split()]
                except:
                    pass
                l = next(f)
            while rez[nset][-1] == 0:
                rez[nset].pop()
            continue
        try:
            l = next(f)
        except:
            break
    f.close()
    return rez


class boundBox(object):
    def __init__(self, xmin: float = 0, xmax: float = 1, ymin: float = 0, ymax: float = 1,
                 zmin: float = 0, zmax: float = 1):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def isPointInside(self, x, y, z):
        rez = x >= self.xmin and x <= self.xmax
        rez = rez and y >= self.ymin and y <= self.ymax
        rez = rez and z >= self.zmin and z <= self.zmax
        return rez

    @property
    def a(self):
        return self.xmax-self.xmin

    @property
    def b(self):
        return self.ymax-self.ymin

    @property
    def c(self):
        return self.zmax-self.zmin

    def __repr__(self):
        return f"""Model bbox:
\tx: {self.xmin}..{self.xmax}
\ty: {self.ymin}..{self.ymax}
\tz: {self.zmin}..{self.zmax}
"""


class node(object):
    __slots__ = ['n', 'x', 'y', 'z']

    def __init__(self, n=1, crds=[0.0, 0.0, 0.0]):
        self.n = n
        self.x = crds[0]
        self.y = crds[1]
        self.z = crds[2]

    def get_crds(self):
        return [self.x, self.y, self.z]

    def set_crds(self, data):
        self.x = data[0]
        self.y = data[1]
        self.z = data[2]

    crds = property(get_crds, set_crds)

    def distTo(self, x, y, z):
        return sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)

    def isInsideBox(self, box):
        return box.isPointInside(self.x, self.y, self.z)


class element(object):
    __slots__ = ['n', '_nodes', 'part', 'etype']

    def __init__(self, n=1, nodes=[], part=1, etype='*ELEMENT'):
        self.n = n
        while 0 in nodes:
            nodes.remove(0)
        self._nodes = nodes
        self.part = part
        self.etype = etype

    def getNodesCount(self):
        return len(set(self.uniqNodes))

    nodesCount = property(getNodesCount)

    def getUniqNodes(self):
        nodes = self._nodes
        for nn in nodes:
            while nodes.count(nn) - 1:
                nodes.remove(nn)
        return nodes

    uniqNodes = property(getUniqNodes)

    def getFaces(self):
        nodes = self.uniqNodes
        if self.etype == "*ELEMENT_SOLID":
            if self.nodesCount == 8:
                finds = [[1, 4, 3, 2],
                         [5, 6, 7, 8],
                         [2, 3, 7, 6],
                         [1, 5, 8, 4],
                         [3, 4, 8, 7],
                         [1, 2, 6, 5]]
            if self.nodesCount == 6:
                finds = [[1, 4, 3, 2],
                         [1, 5, 6, 4],
                         [2, 3, 6, 5],
                         [1, 2, 5],
                         [4, 6, 3]]
            if self.nodesCount == 4:
                finds = [[3, 2, 1],
                         [2, 4, 1],
                         [3, 4, 2],
                         [1, 4, 3]]
            if self.nodesCount == 10:
                finds = [[3, 2, 1, 6, 5, 7],
                         [2, 4, 1, 8, 10, 5],
                         [3, 4, 2, 9, 8, 6],
                         [1, 4, 3, 10, 9, 7]]

        if self.etype == "*ELEMENT_SHELL":
            if self.nodesCount == 4:
                finds = [[1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 1]]
            if self.nodesCount == 3:
                finds = [[1, 2],
                         [2, 3],
                         [3, 1]]
            if self.nodesCount == 6:
                finds = [[1, 2, 4],
                         [2, 3, 5],
                         [3, 1, 6]]
        rez = []
        for idx in finds:
            rez.append([nodes[i - 1] for i in idx])
        return rez

    def getEdges(self):
        nodes = self.uniqNodes
        if self.etype == "*ELEMENT_SOLID":
            if self.nodesCount == 8:
                finds = [[1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 1],
                         [5, 6],
                         [6, 7],
                         [7, 8],
                         [8, 5],
                         [1, 5],
                         [2, 6],
                         [3, 7],
                         [4, 8]]
            if self.nodesCount == 6:
                finds = [[1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 1],
                         [5, 6],
                         [1, 5],
                         [2, 5],
                         [3, 6],
                         [4, 6]]
            if self.nodesCount == 4 or self.nodesCount == 10:
                finds = [[1, 2],
                         [2, 3],
                         [3, 1],
                         [1, 4],
                         [2, 4],
                         [3, 4]]
        if self.etype == "*ELEMENT_SHELL":
            if self.nodesCount == 4:
                finds = [[1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 1]]
            if self.nodesCount == 3 or self.nodesCount == 6:
                finds = [[1, 2],
                         [2, 3],
                         [3, 1]]
        rez = []
        for idx in finds:
            rez.append([nodes[i - 1] for i in idx])
        return rez

    def getEdgesAngles(self):
        nodes = self.uniqNodes
        if self.etype == "*ELEMENT_SOLID":
            if self.nodesCount == 8:
                finds = [[1, 9], [1, -4], [9, -4],
                         [-1, 2], [2, 10], [-1, 10],
                         [-2, 3], [-2, 11], [3, 11],
                         [-3, 12], [4, 12], [-3, 4],
                         [-9, 5], [5, -8], [-8, -9],
                         [-5, 6], [6, -10], [-5, -10],
                         [-6, 7], [-6, 11], [7, -11],
                         [-7, 8], [8, -12], [-7, -12]]
            if self.nodesCount == 6:
                finds = [[1, 6], [1, -4], [6, -4],
                         [-1, 2], [2, 7], [-1, 7],
                         [-2, 3], [-2, 8], [3, 8],
                         [-3, 9], [4, 9], [-3, 4],
                         [-6, -7], [5, -7], [5, -6],
                         [-9, -8], [-5, -8], [-5, -9]]
            if self.nodesCount == 4 or self.nodesCount == 10:
                finds = [[1, -3], [1, 4], [4, -3],
                         [-1, 2], [5, 2], [-1, 5],
                         [-2, 3], [-2, 6], [3, 6],
                         [-4, -5], [-5, -6], [-6, -4]]
        if self.etype == "*ELEMENT_SHELL":
            if self.nodesCount == 4:
                finds = [[1, -4],
                         [-1, 2],
                         [-2, 3],
                         [4, -3]]
            if self.nodesCount == 3 or self.nodesCount == 6:
                finds = [[1, -3],
                         [2, -1],
                         [3, -2]]
        return finds

    def _getNodes(self):
        rez = self._nodes
        if self.etype == '*ELEMENT_SOLID':
            if len(rez) == 4:
                rez += [rez[-1]] * 4
            if len(rez) == 6:
                rez.insert(-1, rez[-2])
                rez.append(rez[-1])
        if self.etype == '*ELEMENT_SHELL':
            if len(rez) == 3:
                rez += [rez[-1]]
        return rez

    def _setNodes(self, nodes):
        self._nodes = nodes

    nodes = property(_getNodes, _setNodes)

    def containsNds(self, nds=1):
        nodes = self.uniqNodes
        if type(nds) == int:
            return nds in nodes
        rez = True
        for n in nds:
            rez = rez and (n in nodes)
        return rez

    def getFaceNum(self, face):
        if not self.containsNds(face):
            return 0
        allFaces = self.getFaces()
        face = set(face)
        rez = 0
        for i, f in enumerate(allFaces):
            if face == set(f):
                rez = i + 1
                break
        return rez


class lsdyna_model:
    __slots__ = ['kwds', 'nodes', 'solids', 'shells', 'progress', 'nodesets']

    def __init__(self, fname=None, procIncludes=False, progress=False):
        """
        :param fname: имя k-файла с задачей LS-DYNA
        :param procIncludes: Если True, то обрабатываются файлы из карты *INCLUDE
        :param progress: Показывать прогресс выполнения длительных задач
        """
        self.kwds = defaultdict(list)
        self.nodes: dict[int, dict[int,node]] = {}
        self.solids: dict[int, dict[int, element]] = {}
        self.shells: dict[int, dict[int, element]] = {}
        self.nodesets: dict[int, list[int]] = {}
        self.progress = progress
        if fname:
            self.read_data(fname, procIncludes)
            self.proc_mesh()
            print(self)

    def read_data(self, fname, includes=False):
        """
        Метод считывает информацию из k-файла с учетом вложений.

        :param fname: Имя k-файла
        :param includes: Если True, то обрабатываются файлы из карты *INCLUDE
        :return:
        """
        self.read_one_file(fname)
        if includes:
            while self.kwds['*INCLUDE']:
                ff = self.kwds['*INCLUDE'].pop()
                for _ in ff.split('\n')[:-1]:
                    if _[0] == '$':
                        continue
                    self.read_one_file(os.path.join(os.path.dirname(fname), _))
            self.kwds.pop('*INCLUDE')

    def read_one_file(self, fname):
        """
        Метод чтения информации из отдельного файла.

        :param fname: имя файла
        :return:
        """
        print('Reading data from {}...'.format(fname), end='\t', flush=True)
        with open(fname, 'r') as k:
            block = []
            for l in k:
                if l.startswith('*'):
                    self.proc_block(block)
                    block = [l]
                else:
                    block.append(l)
            self.proc_block(block)
        print('Done...')

    def proc_block(self, block):
        """
        Метод разбирает блок ключевого слова и сохраняет
        содержание карточки в словарь, ключом является
        keyword.

        :param block: Массив строк.
        :return:
        """
        if block:
            if block[0][0] == '$':
                pass
            elif block[0].upper().strip() in ['*KEYWORD', '*END']:
                pass
            else:
                self.kwds[block[0].upper().strip()].append(''.join(block[1:]))

    def proc_nodes(self):
        """
        Разбирает карточки *NODE и создает слооварь узлов
        self.nodes={номер_узла: узел, ...}

        :return:
        """
        print('Processing nodes...')
        lines = ''.join(self.kwds['*NODE']).split('\n')
        self.nodes = {}
        cprogress = ConsoleProgress(len(lines)) if self.progress else None
        for i, l in enumerate(lines):
            if not l:
                continue
            if l[0] == '$':
                continue
            if "," in l:
                l += "," * (3 - l.count(","))
                ll = l.split(",")
            else:
                l += " " * (8 + 3 * 16 - len(l))
                ll = [l[0:8], l[8:24], l[24:40], l[40:56]]
            nd = node(int(ll[0]), [float(ll[ii]) for ii in range(1, 4)])
            self.nodes[nd.n] = nd
            if cprogress: cprogress.redraw()
        if cprogress: cprogress.finalize()
        del self.kwds['*NODE']

    def proc_elements(self):
        """
        Разбирает карточки *ELEMENT_SOLID и *ELEMENT_SHELL и создает
        словари вида:
        self.solids (self.shells) = {номер_части: {номер_элемента: элемент, ...}, ...}

        :return:
        """
        self.solids = defaultdict(dict)
        self.shells = defaultdict(dict)
        elements = {'solids': ['*ELEMENT_SOLID', self.solids],
                    'shells': ['*ELEMENT_SHELL', self.shells]
                    }
        for etype in elements:
            lines = ''.join(self.kwds[elements[etype][0]]).split('\n')
            print('Processing {}...'.format(etype))
            cprogress = ConsoleProgress(len(lines)) if self.progress else None
            for i, l in enumerate(lines):
                if not l:
                    continue
                if l[0] == '$':
                    continue
                if "," in l:
                    ll = l.split(",")
                else:
                    ll = l.split()
                tmp = list(map(int, ll))
                el = element(tmp[0], tmp[2:], tmp[1], etype)
                elements[etype][1][tmp[1]][el.n] = el
                if cprogress: cprogress.redraw()
            if cprogress: cprogress.finalize()
            del self.kwds[elements[etype][0]]

    def proc_mesh(self):
        """
        Разбор карточек с узлами и элементами.

        :return:
        """
        self.proc_nodes()
        self.proc_elements()

    def __repr__(self):
        rez = 'Number of keywords: {}'.format(len(self.kwds))
        if self.nodes:
            rez += '\nNumber of nodes: {}'.format(len(self.nodes))
        if self.solids:
            rez += '\nNumber of solid parts: {}'.format(len(self.solids))
            for p, s in self.solids.items():
                rez += '\n\tPart {} consists of {} elements'.format(p, len(s))
        if self.shells:
            rez += '\nNumber of shell parts: {}'.format(len(self.shells))
            for p, s in self.shells.items():
                rez += '\n\tPart {} consists of {} elements'.format(p, len(s))
        return rez + '\n'

    def save_nodes(self, fname, nShift=0, mode='a', end=False):
        print('Saving nodes...')
        if not self.nodes:
            return
        if type(fname) == str:
            f = open(fname, mode)
        else:
            f = fname
        if mode == 'w':
            f.write("*KEYWORD\n")
        f.write('*NODE\n')
        for n, c in self.nodes.items():
            f.write("%8d%16.9e%16.9e%16.9e\n" % tuple([n + nShift] + c.get_crds()))
        if end:
            f.write("*END")
        if type(fname) == str: f.close()

    def save_elements(self, fname, nShift=0, eShift=0, pShift=0, mode='a', end=False):
        print('Saving elements...')
        if type(fname) == str:
            f = open(fname, mode)
        else:
            f = fname
        if mode == 'w':
            f.write("*KEYWORD\n")
        if self.solids:
            f.write('*ELEMENT_SOLID\n')
            for p, ps in self.solids.items():
                for e in ps.values():
                    nds = e.nodes
                    fstr = '%8d' * ((len(nds) + 2)) + '\n'
                    f.write(fstr % tuple([e.n + eShift, e.part + pShift] + [nd + nShift for nd in nds]))
        if self.shells:
            f.write('*ELEMENT_SHELL\n')
            for p, ps in self.shells.items():
                for e in ps.values():
                    nds = e.nodes
                    fstr = '%8d' * ((len(nds) + 2)) + '\n'
                    f.write(fstr % tuple([e.n + eShift, e.part + pShift] + [nd + nShift for nd in nds]))
        if end:
            f.write("*END")
        if type(fname) == str: f.close()

    def save_model(self, fname, nShift=0, eShift=0, pShift=0):
        with open(fname, 'w') as f:
            f.write('*KEYWORD\n')
            for kw in self.kwds:
                for data in self.kwds[kw]:
                    f.write('{}\n'.format(kw))
                    f.write(data)
            self.save_nodes(f, nShift=nShift)
            self.save_elements(f, nShift=nShift, eShift=eShift, pShift=pShift)
            f.write('*END\n')

    @property
    def bbox(self) -> boundBox:
        xmin = 1.0e9
        ymin = 1.0e9
        zmin = 1.0e9
        xmax = -1.0e9
        ymax = -1.0e9
        zmax = -1.0e9
        for n in self.nodes.values():
            xmin = min(xmin, n.x)
            ymin = min(ymin, n.y)
            zmin = min(zmin, n.z)
            xmax = max(xmax, n.x)
            ymax = max(ymax, n.y)
            zmax = max(zmax, n.z)
        return boundBox(xmin, xmax, ymin, ymax, zmin, zmax)

    def proceed_node_sets(self):
        self.nodesets = {}
        for set_block in self.kwds['*SET_NODE_LIST']:
            n = 1
            nodes = []
            for line in set_block.split('\n'):
                if line.startswith("$"):
                    continue
                if n==1:
                    set_number = int(line.split()[0])
                else:
                    nodes += [int(s) for s in line.split() if int(s)!=0]
                n += 1
            if nodes:
                self.nodesets[set_number] = nodes

    def nodesets_to_kwds(self):
        for sn in self.nodesets:
            self.kwds['*SET_NODE_LIST'].append(
                formatSetKeyword(self.nodesets[sn], sn)
            )


if __name__=='__main__':
    model = lsdyna_model('../model/main.k', procIncludes=True)
    model.proceed_node_sets()
    print(model.nodesets)