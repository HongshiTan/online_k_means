import math
import random

import numpy as np

from enum import Enum as enum
from sklearn.neighbors import KDTree as kdtree


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class online_k_means:
    def __init__(self, k_target=30):
        self.state_list = enum('state', ['first','init', 'run'])
        self.state = self.state_list.first

        self.k_target = k_target
        self.k = math.ceil((self.k_target -15) /5)

        print('k_target = %d, k = %d'%(self.k_target, self.k))

        self._counter = 0;


    def __w_star_init(self):
        kd = kdtree(self.C, leaf_size=1)
        dist = []
        for vi in self.C:
            d, ind = kd.query([vi], k=2)
            dist.append(d[0][1])
        sorted_dist =np.sort(dist)
        return sum(sorted_dist[0:10])/2

    def __D(self, v):
        dist = []
        for vi in self.C:
            dist.append(np.linalg.norm(v - vi))
        return min(dist)


    def process(self, v):
    #FSM
        if (self.state == self.state_list.first):
        # first vector
            self.C = v
            self.state = self.state_list.init
        elif (self.state == self.state_list.init):
        # init
            kd = kdtree(self.C, leaf_size=1)
            d, ind = kd.query(v, k=1)
            if isclose(d[0][0], 0.0):
                return

            self.C = np.vstack((self.C, v))
            if (self._counter == (self.k  + 10 )):
                self.state = self.state_list.run
                self.r = 1
                self.qr = 0
                self.f = self.__w_star_init()

        # run
        elif (self.state == self.state_list.run):
            r = random.uniform(0,1)
            p = min([self.__D(v)**2/self.f, 1.0])
            if (r < p):
                # selected
                self.C = np.vstack((self.C, v))
                self.qr += 1
            if (self.qr >= self.k):
                self.qr = 0;
                self.f = self.f * 10

        self._counter += 1

    def output(self):
        print(self.C.shape)
        return self.C

