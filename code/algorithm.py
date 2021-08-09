import math
import numpy as np
import random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt

def G(instance,prediction=None):
    obj_value = 0
    for item in instance.item_list:
        if item+obj_value <=instance.capacity:
            obj_value+=item

    return 1.0*obj_value/instance.capacity

def RG(instance,prediction=None):
    bin1 = 0
    bin2 = 0

    for item in instance.item_list:
        if item+bin1 <= instance.capacity:
            bin1 += item
        elif item+bin2 <=instance.capacity:
            bin2 += item


    return (bin1+bin2)*0.5/instance.capacity


def DLA(instance,prediction=None):
    if prediction == None:
        prediction = max(instance.solution_list)

    obj_value = 0



    if prediction <= instance.capacity/2.0:
        for item in instance.item_list:
            if obj_value + item <= instance.capacity:
                obj_value += item

    else:
        tmp_delta = prediction-instance.capacity/2.0
        for item in instance.item_list:
            if item >= tmp_delta and obj_value + item <= instance.capacity:
                obj_value += item

    return 1.0*obj_value/instance.capacity

def RLA(instance,prediction=None):
    delta = 0.5
    if prediction == None:
        prediction = max(instance.solution_list)
    obj_value = 0


    if prediction == 0:
        bin1 = 0
        bin2 = 0

        for item in instance.item_list:
            if item+bin1 <= instance.capacity:
                bin1 += item
            elif item+bin2 <=instance.capacity:
                bin2 += item

        obj_value += (bin1+bin2)*0.5
    elif prediction <= instance.capacity/(2*(delta+1)):
        for item in instance.item_list:
            if obj_value + item <= instance.capacity:
                obj_value += item

    else:
        obj_value_1 = 0
        obj_value_2 = 0
        bin1 = 0
        bin2 = 0
        for item in instance.item_list:
            if item == prediction:
                if obj_value_1 + item <= instance.capacity:
                    obj_value_1 += item
                if obj_value_2 + item <= instance.capacity:
                    obj_value_2 += item

            elif item < prediction:
                if bin1 + item <= instance.capacity-prediction and obj_value_1 + item <= instance.capacity:
                    bin1 += item
                    obj_value_1 += item
                elif bin2 + item <= instance.capacity-prediction and obj_value_2 + item <= instance.capacity:
                    bin2 += item
                    obj_value_2 += item

        obj_value = (obj_value_1+obj_value_2)*0.5

    return 1.0*obj_value/instance.capacity

def RRLA(instance,prediction=None):
    if prediction == None:
        prediction = max(instance.solution_list)


    obj_value = 0
    if prediction == 0:
        bin1 = 0
        bin2 = 0

        for item in instance.item_list:
            if item+bin1 <= instance.capacity:
                bin1 += item
            elif item+bin2 <=instance.capacity:
                bin2 += item

        obj_value += (bin1+bin2)*0.5

    elif prediction <= instance.capacity*0.5:


        #obj_value = RG(instance)
        for item in instance.item_list:
            if obj_value + item <= instance.capacity:
                obj_value += item

    else:
        potential_item_list = []

        for item in instance.item_list:
            if item not in potential_item_list and item <= prediction and item >instance.capacity*0.5:
                potential_item_list.append(item)

        potential_item_list.sort(reverse=True)
        potential_item_list.append(instance.capacity*0.5)
        for index in range(len(potential_item_list)-1):
            flag = 0
            threshold = potential_item_list[index]
            tmp_obj = 0
            for item in instance.item_list:
                if item >= threshold and tmp_obj + item <= instance.capacity:
                    flag = 1
                    tmp_obj += item
                elif flag == 1 and tmp_obj + item <= instance.capacity:
                    tmp_obj += item

            tmp_obj = tmp_obj * (threshold-potential_item_list[index+1])
            obj_value += tmp_obj



        def tmp_random_greedy_algo(instance):
            bin1 = 0
            bin2 = 0

            for item in instance.item_list:
                if item + bin1 <= instance.capacity:
                    bin1 += item
                elif item + bin2 <= instance.capacity:
                    bin2 += item


            return (bin1 + bin2)*0.5
        obj_value_less_half = instance.capacity*0.5*tmp_random_greedy_algo(instance)
        obj_value += obj_value_less_half
        obj_value = obj_value/prediction

    instance.re_init()
    return 1.0*obj_value/instance.capacity








