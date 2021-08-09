
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data_utils import *
import algorithm as al
import pandas as pd
import logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')





EPOCH = 50000#20000

REPORT_FREQ = 400

DOWNLOAD_MNIST = True

HIDDEN_SIZE = 32
NUM_ITEMS = 8
ITEM_RANGE = 20
LR = 0.001
BASE_LINE = 1.0
RANDOM_FACTOR=0.96
BATCH_SIZE = 5
NUM_INSTANCES = 10000

TEACHING_FORCE = 1.0


NUM_INSTANCE_PER_OUTPUT=4

ALGO_NAME = ['G','RG','DLA','RLA','RRLA']





class RNN(nn.Module):
    def __init__(self,input_size=ITEM_RANGE,hidden_size=HIDDEN_SIZE,num_layers=1):
        super(RNN, self).__init__()

        self.batch_size = BATCH_SIZE
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)


        self.out2 = nn.Linear(hidden_size, ITEM_RANGE)

    def forward(self, x,hidden_state=None):

        embedded = self.embedding(x).view(self.batch_size,-1)

        embedded = F.leaky_relu(embedded,0.01)

        r_out, h_s = self.rnn(embedded.view(self.batch_size,1, -1), hidden_state)

        out = self.out2(r_out[:, -1, :])


        return out,h_s




class Adversary:
    def __init__(self,input_size=ITEM_RANGE,hidden_size=HIDDEN_SIZE,num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_layers
        self.model = RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(EPOCH+1),eta_min=0.0001)

    def select_item(self,state,rf,hidden_state=None,min_reward_item=-10):
        probs,h_s = self.model(Variable(state).cuda(),hidden_state)
        probs = F.softmax(probs)
        probs = probs.clamp(min=0.1/ITEM_RANGE,max=0.9)

        random_factor = np.random.uniform(0,1)

        item = probs.multinomial(1).data

        if random_factor >= rf:

            item = torch.tensor([[np.random.randint(0,self.input_size)] for _ in range(BATCH_SIZE)]).cuda()





        if min_reward_item >-10:
            item[-1][0] = min_reward_item
        prob = probs.gather(dim=-1,index=item).view(1,-1)

        log_prob = prob.log()

        entropy = -(probs*probs.log()).sum()

        return item, log_prob,entropy,h_s

    def update_parameters(self,reward,extra_reward,log_probs,entropies,gamma=0):
        self.optimizer.zero_grad()
        loss = 0
        reward = [r*10 for r in reward]
        #print(reward)
        #R = [torch.Tensor(x) for x in reward]#torch.Tensor(reward)
        reward = [torch.Tensor(x) for x in reward]
        extra_reward = torch.Tensor(extra_reward)

        assert len(reward) == NUM_INSTANCE_PER_OUTPUT, 'rewardlen = {0}'.format(len(reward))
        gamma = 0.0

        for i in range(NUM_ITEMS*NUM_INSTANCE_PER_OUTPUT):
            #R = gamma*R + reward[i//NUM_ITEMS] + extra_reward[i]
            R = reward[i//NUM_ITEMS] + extra_reward[i]
            loss = loss - (log_probs[i]*Variable(R).expand_as(log_probs[i]).cuda()).sum() \
                    -(0.0001*entropies[i].cuda()).sum()
        loss = loss/(NUM_ITEMS*BATCH_SIZE)

        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(),5)
        self.optimizer.step()

        #self.scheduler.step()
        return loss

def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad == None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

def draw_ratio(x_list,y_list,fname='Competitive ratio.jpg'):
    plt.cla()
    color_list = ['r',  'y', 'g', 'b', 'k','m']

    line_style = [ '-','--','-.',':',(0, (3, 1, 1, 1, 1, 1))]
    plt.xlabel('Number of Epochs')
    plt.ylabel('Competitive Ratio')

    new_rank_list = [2,3,4,0,1]
    for i in new_rank_list:
        plt.plot(x_list, y_list[i], color=color_list[i],
                 linestyle=line_style[i], linewidth=1,
                 label="{}".format(ALGO_NAME[i]))

    plt.legend()
    plt.savefig(fname, dpi=400)

def train():
    seed = 0
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)

    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.deterministic = True

    performance_list = [ np.zeros(EPOCH//REPORT_FREQ+1) for _ in range(len(ALGO_NAME)) ]


    for iter_id in range(len(ALGO_NAME)):
        baseline = BASE_LINE
        logging.info("--------start algo {}---------".format(ALGO_NAME[iter_id]))

        a_name = ALGO_NAME[iter_id]
        adversary = Adversary()

        min_reward = 100
        min_reward_item_list = []
        min_reward_instance_set = []

        def create_data(LAST=0):
            instance_set = []
            logging.info("-----------Start Creating Data Set for Algo {}".format(ALGO_NAME[iter_id]))

            for _ in range(NUM_INSTANCES//(BATCH_SIZE*NUM_INSTANCE_PER_OUTPUT)):
                state = torch.tensor([[np.random.randint(0, ITEM_RANGE)] for _ in range(BATCH_SIZE)])
                hidden_state = None
                entropies = []
                log_probs = []


                batch_item_list = [[] for _ in range(BATCH_SIZE)]
                rf = 1.1
                for item_index in range(NUM_ITEMS*NUM_INSTANCE_PER_OUTPUT):

                    item, log_prob, entropy, hidden_state = adversary.select_item(state, rf, hidden_state,min_reward_item_list[item_index]-1)
                    hidden_state = hidden_state.cuda()

                    log_probs.append(log_prob)

                    entropies.append(entropy)


                    for batch_id, [x] in enumerate(item.cpu().numpy().tolist()):

                        batch_item_list[batch_id].append(x + 1)

                    state = item



                for batch_id in range(BATCH_SIZE):
                    solution_size = int(NUM_ITEMS * 0.4)

                    solution_size = int(NUM_ITEMS*0.4)

                    tmp_item_set = [ batch_item_list[batch_id][i*NUM_ITEMS:(i+1)*NUM_ITEMS] for i in range(NUM_INSTANCE_PER_OUTPUT) ]


                    solution_list_set = [ x[-solution_size:] for x in tmp_item_set ]


                    for i in range(NUM_INSTANCE_PER_OUTPUT):
                        capacity =sum(solution_list_set[i])
                        instance = Instance(capacity,tmp_item_set[i],solution_list_set[i])
                        assert len(tmp_item_set[i])==NUM_ITEMS,instance.show()
                        instance_set.append(instance)





            write_instance_csv(instance_set,'Algo_{0}_{1}'.format(ALGO_NAME[iter_id],LAST))

        for epoch in range(EPOCH+1):

            state = torch.tensor([[np.random.randint(0,ITEM_RANGE)] for _ in range(BATCH_SIZE)])
            hidden_state = None
            entropies = []
            log_probs = []


            batch_item_list = [ [] for _ in range(BATCH_SIZE) ]
            rf = RANDOM_FACTOR+(1-RANDOM_FACTOR)*epoch/(EPOCH*0.5)
            tf = TEACHING_FORCE + (1 - TEACHING_FORCE) * epoch / (EPOCH * 0.5)
            for item_index in range(NUM_ITEMS*NUM_INSTANCE_PER_OUTPUT):

                if len(min_reward_item_list) >0:
                    item, log_prob, entropy,hidden_state = adversary.select_item(state,rf,hidden_state,min_reward_item_list[item_index]-1)
                else:
                    item, log_prob, entropy, hidden_state = adversary.select_item(state, rf, hidden_state)
                hidden_state = hidden_state.cuda()

                log_probs.append(log_prob)

                entropies.append(entropy)


                for batch_id,[x] in enumerate(item.cpu().numpy().tolist()):

                    batch_item_list[batch_id].append(x+1)



                tmp = np.random.uniform(0,1)
                if len(min_reward_item_list) > item_index and tmp > tf:


                    state = min_reward_item_list[item_index] - 1



                    state = torch.tensor([[state] for _ in range(BATCH_SIZE)])


                else:
                    state = item



            reward_list = []
            reward_mean_list = []

            for batch_id in range(BATCH_SIZE):

                    solution_size = int(NUM_ITEMS*0.4)

                    tmp_item_set = [ batch_item_list[batch_id][i*NUM_ITEMS:(i+1)*NUM_ITEMS] for i in range(NUM_INSTANCE_PER_OUTPUT) ]


                    solution_list_set = [ x[-solution_size:] for x in tmp_item_set ]

                    tmp_instance_set = []
                    for i in range(NUM_INSTANCE_PER_OUTPUT):
                        capacity =sum(solution_list_set[i])
                        instance = Instance(capacity,tmp_item_set[i],solution_list_set[i])
                        assert len(tmp_item_set[i])==NUM_ITEMS,instance.show()
                        tmp_instance_set.append(instance)




                    tmp_reward_list = []
                    tmp_reward_mean_list = []
                    for prediction in range(ITEM_RANGE+1):
                        tmp_reward = np.array([ getattr(al, a_name)(instance,prediction) for instance in tmp_instance_set ])

                        tmp_reward_list.append(tmp_reward)
                        tmp_reward_mean_list.append(np.mean(tmp_reward))

                    reward_mean = max(tmp_reward_mean_list)
                    reward_index  = tmp_reward_mean_list.index(reward_mean)


                    reward_list.append(tmp_reward_list[reward_index])

                    reward_mean_list.append(reward_mean)


                    if reward_mean < min_reward:

                        min_reward = 1.0*reward_mean

                        min_reward_item_list = batch_item_list[batch_id].copy()

                        min_reward_instance_set = tmp_item_set.copy()


            reward_list = np.array(reward_list)
            if baseline > sum(reward_mean_list) / BATCH_SIZE:
                baseline = min(baseline,sum(reward_mean_list) / BATCH_SIZE)
                logging.info('-----------baseline = {}------------'.format(baseline))
                create_data()

            minline = np.min(reward_list)
            maxline = np.max(reward_list)
            if epoch % REPORT_FREQ == 0:
                performance_list[iter_id][epoch // REPORT_FREQ] = sum(reward_mean_list) / BATCH_SIZE
            real_reward_list = reward_list.copy()
            base = 1
            if maxline == minline:
                for i in range(reward_list.shape[0]):
                    for j in range(reward_list.shape[1]):
                        reward_list[i][j] = baseline - reward_list[i][j] + np.random.uniform(-0.5,0.5)


            else:
                if baseline > minline:

                    base = 1.0*sum(reward_mean_list) / BATCH_SIZE
                else:
                    base = 1.0*sum(reward_mean_list) / BATCH_SIZE

                for i in range(reward_list.shape[0]):
                    for j in range(reward_list.shape[1]):
                        reward_list[i][j] = (base - reward_list[i][j])/(maxline-minline)



            new_reward_list = reward_list.T
            extra_reward_list = []
            extra_factor = 0*10
            for i_index in range(NUM_ITEMS*NUM_INSTANCE_PER_OUTPUT):
                tmp_reward_list = []
                for b_index in range(BATCH_SIZE):
                    if batch_item_list[b_index][i_index] == min_reward_item_list[i_index]:
                        tmp_reward_list.append(extra_factor)
                    else:
                        tmp_reward_list.append(0)
                extra_reward_list.append(tmp_reward_list)


            lr = adversary.scheduler.get_lr()[0]
            loss = adversary.update_parameters(new_reward_list, extra_reward_list,log_probs, entropies)
            if epoch % REPORT_FREQ == 0 or epoch == EPOCH:



                log_string = 'Algo {}'.format(ALGO_NAME[iter_id])
                log_string += " epoch {0}".format(epoch)
                log_string += " lr = {}".format(lr)
                log_string += " baseline {:<8.4f}".format(baseline)
                log_string += " base {:<8.4f}\n".format(base)
                for rewards in real_reward_list[-10:]:

                    for reward in rewards:
                        log_string += " real reward {:<8.4f}".format(reward)
                    log_string += '\n'
                reward = np.min(reward_list)
                log_string += " min virtual reward {:<8.4f}\n".format(reward)

                reward = np.max(reward_list)
                log_string += " max virtual reward {:<8.4f}\n".format(reward)

                log_string += " loss={:<8.6f}".format(loss.item())
                log_string += " |g|={}".format(compute_grad_norm(adversary.model))
                logging.info(log_string)

                log_string = 'min reward = {:<8.4f}'.format(min_reward)
                log_string += ' item list = {}'.format(min_reward_instance_set)

                logging.info(log_string)








        create_data(LAST=1)


        print(performance_list)


    x_list = np.arange(0,EPOCH+1,REPORT_FREQ)



    draw_ratio(x_list,performance_list)

def write_instance_csv(instance_set, fname):
    items_fname = '{}_item.csv'.format(fname)
    capacity_fname = '{}_capacity.csv'.format(fname)
    opt_fname = '{}_opt.csv'.format(fname)

    item_dict = {}
    capacity_list = []
    opt_dict = {}
    for index in range(NUM_INSTANCES):
        column = 'I{}'.format(index)
        item_dict[column] = instance_set[index].item_list


        capacity_list.append(instance_set[index].capacity)
        opt_dict[column] = instance_set[index].solution_list



    items_df = pd.DataFrame(item_dict)
    items_df.to_csv(items_fname, index=False,sep=',')

    capacity_df = pd.Series(capacity_list)
    capacity_df.to_csv(capacity_fname,index=False,sep=',')

    opt_df = pd.DataFrame(opt_dict)
    opt_df.to_csv(opt_fname,index=False,sep=',')








if __name__ == '__main__':
    
    train()