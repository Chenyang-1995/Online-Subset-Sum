import math
import numpy as np
import random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import *

import algorithm as al

from train_adversary import NUM_ITEMS

from train_adversary import ITEM_RANGE

import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


ALGO_NAME = ['G','RG','DLA','RLA','RRLA']

def train_predictor(algo_name,instance_set,train_ratio):
    obj_value_list = [0 for _ in range(ITEM_RANGE+1)]

    num_total_instances = len(instance_set)

    num_train_instances = int(num_total_instances*train_ratio)

    num_train_instances_index_list = list(np.random.choice(num_total_instances,num_train_instances,replace=False))

    for i in range(ITEM_RANGE+1):
        prediction = i #+ 1

        for instance_index in num_train_instances_index_list:#range(num_train_instances):
            tmp_instance = instance_set[instance_index]
            obj_value_list[i] += getattr(al,algo_name)(tmp_instance,prediction)


    max_value = max(obj_value_list)
    index_list = [i for i,x in enumerate(obj_value_list) if x >= max_value]
    return max(index_list)


def draw_ratio(x_list,y_list,fname='Competitive ratio over train ratio.jpg'):
    plt.cla()
    color_list = ['r',  'y', 'g', 'b', 'k','m']
    #marker_list = ['o', 'v', '^', '<', '>', 's']
    line_style = [ '-','--','-.',':',(0, (3, 1, 1, 1, 1, 1))]
    plt.xlabel('Number of Training Instances')
    plt.ylabel('Competitive Ratio')
    new_rank_list = [2, 3, 4, 0, 1]
    for i in new_rank_list:
        plt.plot(x_list, y_list[i], color=color_list[i],
                 linestyle=line_style[i], linewidth=1,
                 label="{}".format(ALGO_NAME[i]))
    #plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.legend( loc='lower left', borderaxespad=0)
    #plt.legend()
    plt.savefig(fname, dpi=400)

def draw_ratio_error_bar(x_list,y_list,std_list,fname='Competitive ratio over train ratio.jpg'):
    plt.cla()
    color_list = ['r',  'y', 'g', 'b', 'k','m']
    #marker_list = ['o', 'v', '^', '<', '>', 's']
    line_style = [ '-','--','-.',':',(0, (3, 1, 1, 1, 1, 1))]
    plt.xlabel('Number of Training Instances')
    plt.ylabel('Competitive Ratio')
    new_rank_list = [2, 3, 4, 0, 1]

    fake_x_list = [r'$2^{'+str(x)+'}$' for x in x_list]

    for i in new_rank_list:
        plt.errorbar(x_list, y_list[i], yerr=np.array(std_list[i])*0.5, ecolor=color_list[i],fmt='none')
        plt.plot(x_list, y_list[i], color=color_list[i],
                 linestyle=line_style[i], linewidth=1,
                 label="{}".format(ALGO_NAME[i]))
    #plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xticks(x_list,fake_x_list)
    plt.legend( loc='lower right', borderaxespad=0)
    #plt.legend()
    plt.savefig(fname, dpi=400)

def instance_random_sample(num_instance=10000):
    instance_set = []
    for _ in range(num_instance):
        item_list = [np.random.randint(1,21) for _ in range(8)]
        solution_list = item_list[-3:]
        capacity = sum(solution_list)
        instance_set.append(Instance(capacity,item_list,solution_list))
    return instance_set

def main(repeat=10):

    solution_size = int(NUM_ITEMS*0.4)

    max_log = int(math.log(10000,2))



    train_ratio_list = [0.0001*math.pow(2,x) for x in range(max_log+1)] # + [1]



    trained_obj_value_list = [ [[] for _ in range(len(train_ratio_list))] for _ in range(5) ]
    mean_list = [ [0 for _ in range(len(train_ratio_list))] for _ in range(5) ]
    std_list = [ [0 for _ in range(len(train_ratio_list))] for _ in range(5) ]

    for algo_index,algo_name in enumerate(ALGO_NAME):

        logging.info('--------_Start Algo {}--------'.format(algo_name))




        if algo_name in ['G','RG']:

            csv_fname = 'Algo_{}_0_item.csv'.format(algo_name)

            data = read_data(csv_fname)

            num_intances = data.shape[1]
            logging.info('Num instances: {}'.format(num_intances))

            instance_set = []

            for instance_index in range(num_intances):
                instance_name = 'I{}'.format(instance_index)

                items = data[instance_name].tolist()

                solution = items[-solution_size:]

                capacity = sum(items[-solution_size:])

                tmp_instance = Instance(capacity, items, solution)

                instance_set.append(tmp_instance)

            num_test_instances=len(instance_set)
            tmp_ratio_list = []
            for _ in range(repeat):
                obj_value = 0
                test_instance_indice = list(np.random.choice(num_intances,num_test_instances,replace=False))
                for instance_index in test_instance_indice:#range(num_intances):


                    tmp_instance = instance_set[instance_index]

                    obj_value += getattr(al, algo_name)(tmp_instance)
                obj_value /= num_test_instances#num_intances
                tmp_ratio_list.append(obj_value)
            for tr_index, train_ratio in enumerate(train_ratio_list):



                trained_obj_value_list[algo_index][tr_index]=tmp_ratio_list#.append(obj_value) #[obj_value for _ in range(repeat)]

                mean_list[algo_index][tr_index] = np.mean( np.array(trained_obj_value_list[algo_index][tr_index]) )
                std_list[algo_index][tr_index] = np.std( np.array(trained_obj_value_list[algo_index][tr_index]) )


            logging.info('--------Algo {0}, Ratio {1}--------'.format(algo_name, obj_value))
        if algo_name in ['DLA','RLA','RRLA']:
            instance_set = []



            csv_fname = 'Algo_{}_0_item.csv'.format(algo_name)

            data = read_data(csv_fname)

            tmp_num_intances = data.shape[1]
            logging.info('Num instances: {}'.format(tmp_num_intances))



            for instance_index in range(tmp_num_intances):
                instance_name = 'I{}'.format(instance_index)

                items = data[instance_name].tolist()

                solution = items[-solution_size:]

                capacity = sum(items[-solution_size:])

                tmp_instance = Instance(capacity, items, solution)

                instance_set.append(tmp_instance)


            #instance_set = instance_random_sample()

            num_intances = len(instance_set)
            num_test_instances=len(instance_set)
            random.shuffle(instance_set)


            for tr_index, train_ratio in enumerate(train_ratio_list):

                for _ in range(repeat):
                    obj_value = 0
                    prediction = train_predictor(algo_name,instance_set,train_ratio)
                    test_instance_indice = list(np.random.choice(num_intances,num_test_instances,replace=False))
                    for instance_index in test_instance_indice:#range(num_intances):


                        tmp_instance = instance_set[instance_index]

                        obj_value += getattr(al, algo_name)(tmp_instance,prediction)

                    obj_value /= num_test_instances#num_intances

                    trained_obj_value_list[algo_index][tr_index].append(obj_value)

                    logging.info('--------Algo {0}, train_ratio {1}, prediction {2}----------- Competitive Ratio {3}--------'.format(algo_name,train_ratio,prediction, obj_value))

                mean_list[algo_index][tr_index] = np.mean( np.array(trained_obj_value_list[algo_index][tr_index]) )
                std_list[algo_index][tr_index] = np.std( np.array(trained_obj_value_list[algo_index][tr_index]) )


    x_list = [ x for x in range(max_log+1)]



    draw_ratio_error_bar(x_list,mean_list,std_list)




if __name__ == '__main__':
    main()

