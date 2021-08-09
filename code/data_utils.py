import math
import numpy as np
import random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt


class Instance:
    def __init__(self,capacity,item_list, solution_list):
        self.capacity = capacity
        self.capacity_copy = capacity
        self.item_list = []
        for item in item_list:

                self.item_list.append(item)

        self.item_list_copy=self.item_list.copy()

        self.solution_list = solution_list
        self.solution_list_copy = self.solution_list.copy()

    def show(self):
        print('capacity = {0}, item list: {1}, solution list: {2}'.format(self.capacity,self.item_list,self.solution_list))

    def re_init(self):
        self.capacity = self.capacity_copy
        self.item_list = self.item_list_copy.copy()
        self.solution_list = self.solution_list_copy.copy()

    def round(self,tmp_delta,delta):
        for s_index in range(len(self.solution_list)):
            item = self.solution_list[s_index]
            if item <= 0.5*self.capacity and item > (0.5-tmp_delta)*self.capacity:
                i_index = self.item_list.index(item)
                self.solution_list[s_index] = (0.5-tmp_delta)*self.capacity
                self.capacity -= item
                self.capacity += self.solution_list[s_index]
                self.item_list[i_index] = (0.5-tmp_delta)*self.capacity

            elif item > 0.5*self.capacity and item < (0.5+delta)*self.capacity:
                i_index = self.item_list.index(item)
                self.solution_list[s_index] = (0.5+delta)*self.capacity
                self.capacity -= item
                self.capacity += self.solution_list[s_index]
                self.item_list[i_index] = (0.5+delta)*self.capacity


    def change(self):
        self.re_init()
        num_items = len(self.item_list)
        max_item = max(self.item_list)

        min_item = -1
        for item_index in range(num_items):
            item = self.item_list[item_index]
            noise = np.random.randint(1-item,max_item)
            if item in self.solution_list:
                opt_index = self.solution_list.index(item)
                self.item_list[item_index] += noise
                self.capacity += noise
                self.solution_list[opt_index] += noise
            else:

                self.item_list[item_index] += noise
                if self.item_list[item_index] < min_item or min_item == -1:
                    min_item=self.item_list[item_index]
        min_item = min(self.solution_list)
        max_item = max(self.solution_list)
        avg_item = sum(self.item_list)/num_items
        avg_item = self.capacity/len(self.solution_list)
        if min_item !=0:
            bad_item = self.capacity -avg_item + 1
            self.item_list.insert(0,bad_item)








def read_data(csv_fname):
    df = pd.read_csv(csv_fname)
    return df

if __name__ == '__main__':
    pass