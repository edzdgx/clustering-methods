#!/usr/bin/env python
# -*- coding: utf-8 -*-
# k-means algorithm for 2D data
import numpy as np
import pandas as pd
import random as rd
from matplotlib import pyplot as plt
'''
1 随机选取k个中心点
2 遍历所有数据，将每个数据划分到最近的中心点中
3 计算每个聚类的平均值，并作为新的中心点
4 重复2-3，直到这k个中线点不再变化（收敛了），或执行了足够多的迭代
'''

class Kmeans:
    def __init__(self, _data, _k):
        self.data = _data   # input data
        self.k = _k         # number of clusters
        self.clusters = []  # clusters of nodes
        self.centers = []   # center of clusters
        self.dist = []      # distance vector

    def center_select(self):
        # max_data = (self.data.max()[0], self.data.max()[1])
        # min_data = (self.data.min()[0], self.data.min()[1])
        # print max_data, min_data
        for i in range(self.k):
            self.centers.append((rd.uniform(self.data.min()[0], self.data.max()[0]),
                          rd.uniform(self.data.min()[1], self.data.max()[1])))
        # print self.centers

    def init(self):
        self.center_select()
        for idx in range(self.k):
            self.clusters.append([])
            self.dist.append([])
        # print self.clusters

        for point in self.data.values:
            # print point
            d = 100000000
            for idx, center in enumerate(self.centers):
                new_d = ((center[0]-point[0])**2+(center[1]-point[1])**2)**0.5
                if new_d < d:
                    d = new_d
                    pos = idx
            self.dist[pos].append(d)
            self.clusters[pos].append(point)
        # print self.dist
        # print self.clusters
        # print self.centers



    def next_iter(self):
        new_centers = []
        # max iteration = 5
        for idx in range(5):
            # calculate average for each cluster center, make them new centers
            for cList in self.clusters:
                # print cList
                tot0 = 0
                tot1 = 0
                for tup in cList:
                    tot0 += tup[0]
                    tot1 += tup[1]
                new_centers.append((tot0/len(cList), tot1/len(cList)))
            # print self.centers
            # print new_centers
            if new_centers == self.centers:
                return self.clusters

            # reset and reinitialize
            self.clusters = []
            self.dist = []
            for idx in range(self.k):
                self.clusters.append([])
                self.dist.append([])

            # calculate dist to new_centers
            for point in self.data.values:
                d = 100000000
                for idx, center in enumerate(new_centers):
                    print(center[0], point[0])
                    new_d = ((center[0]-point[0])**2+(center[1]-point[1])**2)**0.5
                    if new_d < d:
                        d = new_d
                        pos = idx
                self.dist[pos].append(d)
                self.clusters[pos].append(point)

    def k_means(self):
        pass
