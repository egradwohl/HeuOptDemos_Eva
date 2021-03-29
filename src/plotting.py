import sys

from networkx import algorithms
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")
from pymhlib.demos.maxsat import MAXSATInstance
from pymhlib.demos.misp import MISPInstance

import matplotlib.image as mpimg
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import statistics

from .problems import InitSolution, Problem, Algorithm, Option
from abc import ABC, abstractmethod
from .logdata import read_from_logfile, Log
from matplotlib.lines import Line2D
from dataclasses import dataclass
from matplotlib import gridspec
import random as rd
import matplotlib.patches as patches


plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
pc_dir = 'pseudocode'
pc_img_type = '.png'


@dataclass
class CommentParameters:
        opt: Option = Option.LI
        status: str = 'start'
        n: int = 0
        m: int = 0
        par: int = 0
        gain = 0
        better: bool = False
        no_change: bool = False
        best = 0
        obj = 0
        enditer: bool = False

        # algorithm specific parameters
        remove: set = None
        add: set = None
        flip: list = None
        k: int = None
        alpha: float = None
        thres: float = None
        ll: int = 0
        asp: bool = False

class Draw(ABC):

        comments = {}
        grey = str(210/255) #'lightgrey'
        darkgrey = str(125/255)
        white = 'white'
        blue = 'royalblue'
        red = 'tomato'
        green = 'limegreen'
        yellow = 'gold'
        orange = 'darkorange'

        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log, dmin: float):
                self.problem = prob
                self.algorithm = alg
                self.dmin = dmin
                self.graph = self.init_graph(instance)
                self.log_granularity = log_granularity
                self.pc_imgs = self.load_pc_img_files()
                self.legend_elems = self.create_legend_elems()
                # create figure
                plt.close() # close any previously drawn figures
                gs = gridspec.GridSpec(2, 2,height_ratios=[1,8], width_ratios=[2,1] )
                fig = plt.figure(num = f'Solving {prob.value} with {alg.value}')
                self.descr_ax = fig.add_subplot(gs[0, :])
                self.ax = fig.add_subplot(gs[1,0])
                self.img_ax = fig.add_subplot(gs[1,1])
                self.img_ax.axis('off')
                self.ax.axis('off')
                self.descr_ax.axis('off')




        @abstractmethod
        def init_graph(self, instance):
                return None

        def get_animation(self, i: int, log_data: list):
                self.reset_graph()
                comment = CommentParameters()
                if self.algorithm == Algorithm.TS:
                        comment = self.get_ts_animation(i,log_data)
                if self.algorithm == Algorithm.GVNS:
                        comment = self.get_gvns_animation(i,log_data)
                if self.algorithm == Algorithm.GRASP:
                        comment = self.get_grasp_animation(i,log_data)

                self.add_description(i, comment)
                self.add_legend()
                self.load_pc_img(i, log_data, comment)


        @abstractmethod
        def get_grasp_animation(self, i: int, log_data: list):
                pass


        @abstractmethod
        def get_gvns_animation(self, i:int, log_data:list):
                pass

        @abstractmethod
        def get_ts_animation(self, i:int, log_data:list):
                pass

        def add_description(self, i, comment: CommentParameters):
                phase = ''
                if comment.status == 'init':
                        phase = f'{self.problem.value} Instance'
                else:
                        phase = comment.opt.value if comment.opt != Option.TL else Option.LI.value

                if self.log_granularity == Log.Cycle and comment.opt != Option.CH:
                        if self.algorithm == Algorithm.GVNS and i > 2:
                                phase = Option.SH.value + ' + ' + Option.LI.value
                        if self.algorithm == Algorithm.GRASP:
                                phase = Option.RGC.value + ' + ' + Option.LI.value

                text = '\n'.join(
                        ['%s: %s' % (phase, self.create_comment(comment)),
                        'Best Objective: %d' % (comment.best, ),
                        'Current Objective: %d' % (comment.obj, )]
                        )
                self.descr_ax.clear()
                self.descr_ax.axis('off')
                self.descr_ax.text(0,1, text, horizontalalignment='left', verticalalignment='top')#, transform=self.ax.transAxes)


        def add_legend(self):
                self.ax.legend(self.legend_elems[0], self.legend_elems[1],  ncol=2, handlelength=1, borderpad=0.7, columnspacing=0, loc='lower left')

        @abstractmethod
        def create_legend_elems(self):
                pass

        def load_pc_img_files(self):
                filenames = os.listdir(pc_dir + os.path.sep + self.algorithm.name.lower())
                pc_imgs = dict()
                for f in filenames:
                        pc_imgs[f.lower()] = mpimg.imread(pc_dir + os.path.sep + self.algorithm.name.lower() + os.path.sep + f)
                return pc_imgs


        def load_pc_img(self, i: int, log_data: list, comment: CommentParameters):
                if self.algorithm == Algorithm.GVNS:
                        self.load_gvns_pc_img(i, log_data, comment)
                        return
                if self.algorithm == Algorithm.GRASP:
                        self.load_grasp_pc_img(comment)
                        return
                if self.algorithm == Algorithm.TS:
                        self.load_ts_pc_img(comment)
                        return

        def load_gvns_pc_img(self, i, log_data, comment: CommentParameters):
                # find out, if li is part of initial vnd
                init = ''
                if comment.opt == Option.LI and self.log_granularity != Log.Cycle:
                        j = i
                        while j > 0:
                                if log_data[j].get('m','') == 'ch':
                                        init = '_init'
                                        break
                                if log_data[j].get('m','') == 'sh':
                                        break
                                j -= 1
                elif comment.opt == Option.LI and log_data[i-1].get('m','') == 'ch':
                        init = '_init'

                level = self.log_granularity.name.lower() if self.log_granularity != Log.NewInc else Log.StepNoInter.name.lower()
                m = '_' + comment.opt.name.lower()
                status = '_' + comment.status if not comment.enditer else '_enditer'
                better = '_better' if (comment.opt == Option.LI and not comment.no_change and comment.status == 'end' and not comment.enditer) or (init == '' and comment.opt == Option.LI and comment.enditer and comment.better) else ''
                path = level + m + status + init + better + pc_img_type
                img = self.pc_imgs[path]
                self.img_ax.set_aspect('equal', anchor='E')
                self.img_ax.imshow(img)#,extent=[0, 1, 0, 1])

        def load_grasp_pc_img(self, comment: CommentParameters):

                level = self.log_granularity.name.lower() if self.log_granularity != Log.NewInc else Log.StepNoInter.name.lower()
                m = '_' + comment.opt.name.lower()
                status = '_' + comment.status if not comment.enditer else '_enditer'
                better = '_better' if comment.opt != Option.RGC and comment.better else ''
                path = level + m + status + better + pc_img_type
                img = self.pc_imgs[path]
                self.img_ax.set_aspect('equal', anchor='E')
                self.img_ax.imshow(img)

        def load_ts_pc_img(self, comment: CommentParameters):
                
                level = self.log_granularity.name.lower() if self.log_granularity == Log.StepInter else Log.StepNoInter.name.lower()
                m = '_' + comment.opt.name.lower() if comment.opt == Option.CH else '_' + Option.LI.name.lower()
                status = '_' + comment.status if not comment.enditer else '_enditer'
                better = '_better' if comment.opt != Option.CH and comment.better else ''
                path = level + m + status + better + pc_img_type
                img = self.pc_imgs[path]
                self.img_ax.set_aspect('equal', anchor='E')
                self.img_ax.imshow(img)

        @abstractmethod
        def reset_graph(self):
                pass

        @abstractmethod
        def draw_graph(self):
                pass

        def create_comment(self, params: CommentParameters):
                option = params.opt
                status = params.status
                if self.log_granularity == Log.StepInter:
                        return self.comments[option][status](params)
                if status == 'init':
                        return self.comments[option][status](params)
                if self.log_granularity == Log.StepNoInter or self.log_granularity == Log.NewInc:
                        comment = self.comments[option]
                        if option == Option.RGC and not status in ['start', 'end']:
                                return ', '.join([comment.get('cl','')(params),  '\n' + comment.get('rcl','')(params), comment.get('sel','')(params)])
                        return ', '.join([comment['start'](params), comment['end'](params)])

                if self.log_granularity == Log.Update or self.log_granularity == Log.Cycle:
                        comment = self.comments[option]
                        return comment['cycle_start'](params) + comment['end'](params)

        def calculate_node_position(self,graph):
                init_pos = nx.kamada_kawai_layout(graph)
                in_place = False
                i = 0
                while i < 50 and not in_place:
                        in_place = True
                        for n, point1 in init_pos.items():
                                for m, point2 in init_pos.items():
                                        if m==n:
                                                continue
                                        dist = np.linalg.norm(point1 - point2)
                                        dist = max(dist, 0.0001)
                                        if dist < self.dmin:
                                                in_place = False
                                                r = (point1 - point2)/dist
                                                new_pos = point1 + r * self.dmin
                                                new_pos[0], new_pos[1] = max(rd.uniform(-0.85, -1), new_pos[0]), max(rd.uniform(-0.85, -1), new_pos[1])
                                                new_pos[0], new_pos[1] = min(rd.uniform(0.85, 1), new_pos[0]), min(rd.uniform(0.85, 1), new_pos[1])
                                                init_pos[m] = new_pos
                        i +=1
                return init_pos

        

class MISPDraw(Draw):

        comments = {
                        Option.CH:{
                                'init': lambda params: f'{params.n} nodes, {params.m} edges',
                                'start': lambda params: f'construction={InitSolution(params.par).name}, add {len(params.add)} node(s)',
                                'end': lambda params: f'objective gain={params.gain}',
                                'cycle_start': lambda params:  f'construction={InitSolution(params.par).name}, add {len(params.add)} node(s)\n'
                        },
                        Option.LI: {
                                'start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)',
                                'end': lambda params: f'objective gain={params.gain}{", no improvement - reached local optimum" if params.no_change else ""}{", found new best solution" if params.better else ""}',
                                'cycle_start': lambda params: f'remove {len(params.remove)} node(s), add {len(params.add)} node(s)\n'
                        },
                        Option.SH: {
                                'start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)',
                                'end': lambda params: f'objective gain={params.gain}{", found new best solution" if params.better else ""}',
                                'cycle_start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)\n'
                        },
                        Option.RGC:{
                                'start': lambda params: 'start with empty solution',
                                'end': lambda params: f'created complete solution{", found new best solution" if params.better else ""}',
                                'cl': lambda params: 'candidate list=remaining degree (number of unblocked neigbors)',
                                'rcl': lambda params: f'restricted candidate list=' + (f'{params.k}-best' if params.k else f'alpha: {params.alpha}, threshold: {params.thres}'),
                                'sel': lambda params: f'selection=random, objective gain={params.gain}',
                                'cycle_start': lambda params: f'restricted candidate list=' + (f'{params.k}-best\n' if params.k else f'alpha: {params.alpha}\n')
                        },
                        Option.TL:{
                                'start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)',
                                'end': lambda params: f'size of tabu list={params.ll}, objective gain={params.gain}{", applied aspiration criterion" if params.asp else ""}{", all possible exchanges are tabu" if params.no_change else ""}{", found new best solution" if params.better else ""}',
                                'cycle_start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)\n'
                               
                        }
                        
                        }


        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log, dmin: float):
                super().__init__(prob,alg,instance,log_granularity, dmin)

        def init_graph(self, instance):
                graph = instance.graph
                pos = self.calculate_node_position(graph)
                nx.set_node_attributes(graph, {n:{'color':self.grey, 'label':'', 'tabu':False} for n in graph.nodes()})
                nx.set_node_attributes(graph, pos, 'pos')
                nx.set_edge_attributes(graph, self.grey, 'color')
                return graph




        def get_gvns_animation(self, i:int, log_data: list):
                data = log_data[i]
                comment_params = CommentParameters()
                done = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params

                comment_params.opt = Option[data.get('m','li').upper()]
                if comment_params.enditer:
                        j = i-1
                        while j > 0:
                                if log_data[j].get('end_iter',False):
                                        comment_params.better = log_data[i].get('best',0) > log_data[j].get('best',0)
                                        break
                                j -=1
                self.draw_graph(data['inc'])
                return comment_params

        def get_gvns_and_ts_animation(self, i:int, log_data: list, comment_params: CommentParameters):
                data = log_data[i]
                status = data.get('status','')
                sol = data.get('sol',[])
                comment_params.status = status
                comment_params.par = data.get('par',1)
                if status == 'init':# or (status == 'start' and (data.get('m') == 'ch')):
                        comment_params.n = len(self.graph.nodes())
                        comment_params.m = len(self.graph.edges())
                        comment_params.opt = Option.CH
                        comment_params.best = comment_params.obj = 0
                        comment_params.gain = data.get('obj',0)
                        self.draw_graph()
                        return True

                remove,add = self.get_removed_and_added_nodes(i,log_data)
                nx.set_node_attributes(self.graph, {n:self.blue for n in sol}, 'color')
                nx.set_node_attributes(self.graph, {n:{'label':'+','color':self.green if status == 'start' else self.blue} for n in add})
                nx.set_node_attributes(self.graph, {n:{'label':'-','color':self.red if status == 'start' else self.grey} for n in remove})
                nx.set_node_attributes(self.graph, {n:self.yellow for n in data.get('inc') if data.get('better',False)}, 'color')
                nx.set_edge_attributes(self.graph, {e:'black' for n in add for e in self.graph.edges(n)} if status == 'end' else {}, 'color')

                # fill parameters for plot description
                comment_params.remove = remove
                comment_params.add = add
                comment_params.gain =  data.get('obj', 0) if comment_params.opt == Option.CH else data.get("obj",0) - log_data[i-1].get('obj',0)
                comment_params.no_change = not (add or remove)
                comment_params.better = data.get('better',False) or (log_data[i-1].get('best',0)< data.get('best',0))
                comment_params.best = data['best']
                comment_params.obj = data['obj']
                comment_params.enditer = data.get('end_iter',False)
                return False


        def get_removed_and_added_nodes(self, i, log_data):
                status = log_data[i].get('status','start')
                compare_i = i + (status == 'start') - (status == 'end')
                compare_sol = set(log_data[compare_i].get('sol',[]))
                current_sol = set(log_data[i].get('sol',[]))
                remove = current_sol - compare_sol if status == 'start' else compare_sol - current_sol
                add = current_sol - compare_sol if status == 'end' else compare_sol - current_sol
                return remove,add


        def get_grasp_animation(self, i:int, log_data: list):
                if log_data[i].get('m','') in ['ch', 'li'] or log_data[i].get('status','') == 'init':
                        comment = self.get_gvns_animation(i,log_data)
                        if log_data[i].get('end_iter'):
                                j = i-1
                                while j >= 0:
                                        if log_data[j].get('end_iter') or j == 0:
                                                comment.better = log_data[i].get('best',0) > log_data[j].get('best',0)
                                                break
                                        j -= 1
                        return comment

                data = log_data[i] 
                comment_params = CommentParameters()
                comment_params.status = data.get('status','')
                comment_params.opt = Option.RGC
                par = data.get('par',1)
                mn = min(data.get('cl',{0:0}).values())
                mx = max(data.get('cl',{0:0}).values())
                if type(par) == int:
                        comment_params.k = par
                else:
                        comment_params.alpha = par
                        comment_params.thres = round(mn + par * (mx-mn),2)

                if comment_params.status in ['start','end']:
                        comment_params.better = data.get('better',False)
                        comment_params.best = data.get('best',0)
                        comment_params.obj = data.get('obj',0)
                        nx.set_node_attributes(self.graph, {n:self.yellow if data.get('better',False) else self.blue for n in data.get('sol') if comment_params.status == 'end'}, name='color')
                        self.draw_graph(data.get('inc') if comment_params.status == 'end' else [])
                        return comment_params

                nx.set_node_attributes(self.graph, data.get('cl',{}), 'label')
                nx.set_node_attributes(self.graph, {n: self.green if data.get('sel',-1) == n else self.blue for n in data.get('sol', [])}, name='color')
                selected = set() if not 'sel' in data else set(self.graph.neighbors(data.get('sel')))
                n_unsel = selected.intersection(set(data['cl'].keys()))
                nx.set_node_attributes(self.graph, {n:self.orange for n in n_unsel},'color')
                nx.set_edge_attributes(self.graph, {(data.get('sel',n),n):'black' for n in n_unsel}, 'color')

                j = i
                while not (log_data[j]['status'] in ['start','end']):
                        j -= 1
                        if j == 0:
                                break
                comment_params.best = log_data[j].get('best',0)
                comment_params.obj = len(data.get('sol',[]))
                comment_params.gain = 1
                self.draw_graph(data.get('rcl',[]), sel_color='black')
                return comment_params

        def get_ts_animation(self, i:int, log_data: list):

                data = log_data[i]
                comment_params = CommentParameters()
                comment_params.status = data.get('status','')

                done = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params

                tabu_list = data.get('tabu',[])
                asp_nodes = set()
                for ta in tabu_list:
                        tabu_nodes = list(ta[0])
                        life = ta[1]
                        nx.set_node_attributes(self.graph, {n: {'label':life,'tabu':True} for n in tabu_nodes})
                        if comment_params.status == 'start' and set(tabu_nodes).issubset(comment_params.add):
                                asp_nodes = asp_nodes.union(set(tabu_nodes).intersection(comment_params.add))
                        

                comment_params.asp = len(asp_nodes) > 0 or (comment_params.status == 'end' and 'ta_del' in data.keys())
                comment_params.ll = data.get('ll',0)
                comment_params.opt = Option.CH if data.get('m').startswith('ch') else Option.TL
                self.draw_graph(data.get('inc',[]))
                return comment_params


        def create_legend_elems(self):

                legend_elements = [
                        Line2D([0], [0],linestyle='none'),
                        Line2D([0],[0],marker='o', color='w',
                                markerfacecolor=self.red, markersize=13),
                        Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=self.yellow, markersize=13),
                        Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=self.blue, markersize=13),
                        Line2D([0],[0],marker='o', color='w',
                                markerfacecolor=self.green, markersize=13),
                        Line2D([0], [0], marker='o', linestyle='none',
                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11)     
                        ]
                description = ['','','','current solution','remove/add node', 'best solution']

                if self.algorithm == Algorithm.TS:
                        legend_elements.insert(3,(Line2D([0], [0],linestyle='none')))
                        legend_elements.append(Line2D([0],[0],marker='X', color='w',
                                        markerfacecolor='black', markersize=13))
                        description.insert(3,'')
                        description.append('tabu attribute')

                if self.algorithm == Algorithm.GRASP:
                        legend_elements.insert(3,(Line2D([0], [0],linestyle='none')))
                        legend_elements.append(Line2D([0],[0],marker='o', color='w',
                                        markerfacecolor=self.orange, markersize=13))
                        description.insert(3,'')
                        description.append('blocked neighbor')

                return (tuple(legend_elements), tuple(description))

        def reset_graph(self):
                nx.set_node_attributes(self.graph, self.grey, name='color')
                nx.set_edge_attributes(self.graph, self.grey, name='color')
                nx.set_node_attributes(self.graph, '', name='label')
                nx.set_node_attributes(self.graph,False,name='tabu')

        def draw_graph(self, pos_change: list() = [], sel_color='gold'):
                self.ax.clear()
                self.ax.set_ylim(bottom=-1.5,top=1.1)
                self.ax.set_xlim(left=-1.1,right=1.1)
                for pos in ['right', 'top', 'bottom', 'left']: 
                        self.ax.spines[pos].set_visible(False) 

                nodelist = self.graph.nodes()

                pos = nx.get_node_attributes(self.graph,'pos')

                color = [self.graph.nodes[n]['color'] for n in nodelist]
                linewidth = [3 if n in pos_change else 0 for n in nodelist]
                lcol = [sel_color for _ in nodelist]
                labels = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if not self.graph.nodes[v]['tabu']}
                edges = list(self.graph.edges())
                e_cols = [self.graph.edges[e]['color'] for e in edges]
                
                nx.draw_networkx(self.graph, pos, nodelist=nodelist, with_labels=True, labels=labels,font_weight='bold', font_size=14, ax=self.ax,  
                                node_color=color, edgecolors=lcol, edgelist=edges, edge_color=e_cols, linewidths=linewidth, node_size=500)

                # drawings for tabu search
                nodes_labels_tabu = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['tabu']}
                tabu_nodes = list(nodes_labels_tabu.keys())
                if len(tabu_nodes) == 0:
                        return
                x_pos = {k:[v[0]-0.03,v[1]-0.03] for k,v in pos.items()}
                nx.draw_networkx_nodes(self.graph, x_pos, nodelist=tabu_nodes,node_color='black', node_shape='X', node_size=150,ax=self.ax)
                nx.draw_networkx_labels(self.graph, pos, labels=nodes_labels_tabu, ax=self.ax, font_size=12, font_weight='bold', 
                                        font_color='black',horizontalalignment='left',verticalalignment='baseline')






class MAXSATDraw(Draw):

        comments = {
                Option.CH:{
                        'init': lambda params: f'{params.n} variables, {params.m} clauses',
                        'start': lambda params: f'construction={InitSolution(params.par).name}',
                        'end': lambda params: f'objective gain={params.gain}',
                        'cycle_start': lambda params: f'construction={InitSolution(params.par).name}\n'
                },
                Option.LI: {
                        'start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)',
                        'end': lambda params: f'objective gain={params.gain}{", no improvement - reached local optimum" if params.no_change else ""}{", found new best solution" if params.better else ""}',
                        'cycle_start': lambda params: f'flipping {len(params.flip)} variable(s)\n'
                },
                Option.SH: {
                        'start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)',
                        'end': lambda params: f'objective gain={params.gain}{", found new best solution" if params.better else ""}',
                        'cycle_start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)\n'
                },
                Option.RGC:{
                        'start': lambda params: 'start with empty solution',
                        'end': lambda params: f'created complete solution{", found new best solution" if params.better else ""}',
                        'cl': lambda params: 'candidate list=number of additionally fulfilled clauses',
                        'rcl': lambda params: f'restricted candidate list=' + (f'{params.k}-best' if params.k else f'alpha: {params.alpha}, threshold: {params.thres}'),
                        'sel': lambda params: f'selection=random, objective gain={params.gain}',
                        'cycle_start': lambda params: f'restricted candidate list=' + (f'{params.k}-best' if params.k else f'alpha: {params.alpha}') + '\n'
                },
                Option.TL:{
                        'start': lambda params: f'k={params.par} flipping {len(params.flip)} variable(s)',
                        'end': lambda params: f'size of tabu list={params.ll}, objective gain={params.gain}{", applied aspiration criterion" if params.asp else ""}{", all possible flips are tabu" if params.no_change else ""}{", found new best solution" if params.better else ""}',
                        'cycle_start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)\n'
                       
                }
                }


        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log, dmin: float):
                super().__init__(prob,alg,instance,log_granularity, dmin)


        def init_graph(self, instance):

                n = instance.n #variables
                m = instance.m #clauses
                clauses = [i for i in range(1,1+m)]
                variables = [i + m for i in range(1,1+n)]
                incumbent = [i + n for i in variables]

                #sort clauses by barycentric heuristic (average of variable positions)
                '''
                def avg_clause(clause):
                        i = clause-1
                        vl = list(map(abs,instance.clauses[i]))
                        l = len(instance.clauses[i])
                        return sum(vl)/l
                '''
                #sort clauses by median heuristic (median of variable position)
                def median_clause(clause):
                        i = clause-1
                        vl = list(map(abs,instance.clauses[i]))
                        vl.sort()
                        j = len(vl)
                        pos = vl[int(j/2)]
                        return pos

                clauses_sorted = sorted(clauses, key=median_clause)

                ### calculate positions for nodes
                step = 2/(n+1)
                pos = {v:[-1 + i*step, 0.4] for i,v in enumerate(variables, start=1)}
                pos.update({i:[-1 +j*step,0.4+0.2] for j,i in enumerate(incumbent, start=1)})
                step = 2/(m+1)
                pos.update({c:[-1+ i*step,-0.4] for i,c in enumerate(clauses_sorted, start=1)})

                # create nodes with data
                v = [(x, {'type':'variable', 'nr':x-m, 'color':self.grey, 'pos':pos[x], 'label':'','usage':clause,'alpha':1.,'tabu':False}) for x,clause in enumerate(instance.variable_usage, start=m+1)]  #[m+1,...,m+n]
                c = [(x, {'type':'clause', 'nr':x, 'color': self.grey, 'pos':pos[x], 'label':f'c{x}', 'clause':clause}) for x,clause in enumerate(instance.clauses, start=1)]   #[1,..,m]
                i = [(x, {'type':'incumbent', 'nr':x-m-n, 'color':self.white, 'pos':pos[x], 'label':f'x{x-m-n}','alpha':1.}) for x in incumbent]               #[1+m+n,...,2n+m]

                # create graph by adding nodes and edges
                graph = nx.Graph()
                graph.add_nodes_from(c)
                graph.add_nodes_from(v)
                graph.add_nodes_from(i)

                for i,cl in enumerate(instance.clauses, start=1):
                        graph.add_edges_from([(i,abs(x)+ m,{'style':'dashed' if x < 0 else 'solid', 'color':self.grey}) for x in cl])

                graph.__setattr__('ts_reposition', False)

                return graph

        def get_gvns_animation(self, i:int, log_data:list):
                comment_params = CommentParameters()
                data = log_data[i]
                comment_params.status = data.get('status','')
                done, lit_info = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params
                
                flipped_nodes = [] if comment_params.status == 'end' else comment_params.flip
                flipped_nodes += [n for n,t in self.graph.nodes(data='type') if t=='incumbent'] if data.get('better',False) else []
                if comment_params.enditer:
                        j = i-1
                        while j > 0:
                                if log_data[j].get('end_iter',False):
                                        comment_params.better = log_data[i].get('best',0) > log_data[j].get('best',0)
                                        break
                                j -=1
                self.draw_graph(flipped_nodes + list(comment_params.add.union(comment_params.remove)))
                self.write_literal_info(lit_info)
                self.add_sol_description(i,data)
                return comment_params

        def get_gvns_and_ts_animation(self, i:int, log_data: list, comment_params: CommentParameters):

                incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
                variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
                clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']
                data = log_data[i]
                comment_params.status = data.get('status','')
                comment_params.opt = Option[data.get('m','li').upper()]
                if comment_params.status == 'init':#'start' and (data.get('m') == 'ch' or i==0):
                        comment_params.n = len(variables)
                        comment_params.m = len(clauses)
                        comment_params.obj = comment_params.best = 0
                        comment_params.opt = Option.CH
                        nx.set_node_attributes(self.graph,{n:'' for n,t in self.graph.nodes(data='type') if t=='incumbent'}, name='label')
                        self.draw_graph([])
                        return True, {}
                if comment_params.opt == Option.CH and comment_params.status == 'start':
                        comment_params.obj = comment_params.best = 0
                        comment_params.opt = Option.CH
                        comment_params.par = data.get('par',0)

                        nx.set_node_attributes(self.graph, {k: self.grey for k in incumbent}, name='color')
                        self.draw_graph([])
                        self.add_sol_description(i,data)
                        return True, {}
                if comment_params.opt == Option.CH and comment_params.status=='end':
                        log_data[i-1]['sol'] = [-1 for _ in data['sol']]

                nx.set_node_attributes(self.graph, {k: self.red if data['inc'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in incumbent}, name='color')
                nx.set_node_attributes(self.graph, {k: self.red if data['sol'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in variables}, name='color')
                added, removed, pos_literals = self.color_and_get_changed_clauses(i, log_data, comment_params.status == 'start')
                flipped_nodes = self.get_flipped_variables(i,log_data)

                comment_params.flip = flipped_nodes
                comment_params.add = added
                comment_params.remove = removed
                comment_params.par = data.get('par',1)
                comment_params.gain = data.get('obj', 0) if comment_params.opt == Option.CH else data.get("obj",0) - log_data[i-1].get('obj',0)
                comment_params.better = data.get('better',False) or (log_data[i-1].get('best',0) < data.get('best',0))
                comment_params.no_change = len(flipped_nodes) == 0
                comment_params.enditer = data.get('end_iter',False)
                comment_params.best = data.get('best', False)
                comment_params.obj = data.get('obj', False)

                return False, pos_literals
                



        def get_flipped_variables(self, i: int, log_data: list):
                info = log_data[i]
                comp = i + (info['status'] == 'start') - (info['status'] == 'end')
                flipped_variables = [n for n,v in enumerate(info['sol'],start=1) if v != log_data[comp]['sol'][n-1]]
                flipped_variables = [n for n, data in self.graph.nodes(data=True) if data['nr'] in flipped_variables and data['type']=='variable']

                fullfilled_clauses = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'clause' and self.graph.nodes[n]['color']==self.green]
                for c in fullfilled_clauses:
                        for v in self.graph.neighbors(c):
                                col = self.graph.nodes[v]['color']
                                style = self.graph.edges[(v,c)]['style']
                                if (col == self.blue and style=='solid') or (col==self.red and style=='dashed'):
                                        self.graph.edges[(v,c)]['color'] = self.darkgrey
        
                if info['status'] == 'end' and info.get('m','') != 'ch':
                        nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if set(edge) & set(flipped_variables)}, 'color')
                return flipped_variables

        def color_and_get_changed_clauses(self,value,log_data, start=True):

                i = value - (not start)
                pos_start,literals = self.color_clauses_and_count_literals(log_data[i], literals=start)

                if start:
                        return set(),set(),literals

                pos_end,literals = self.color_clauses_and_count_literals(log_data[value])

                return pos_end-pos_start, pos_start-pos_end,literals

        def color_clauses_and_count_literals(self, log_data: dict, literals=True):

                clauses = nx.get_node_attributes(self.graph, 'clause')
                fulfilled = set()
                num_literals = dict()

                for n,clause in clauses.items():
                        for v in clause:
                                if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                        fulfilled.add(n)
                                        self.graph.nodes[n]['color'] = self.green
                                        break
                                else:
                                        self.graph.nodes[n]['color'] = self.grey

                if literals:
                        num_literals = dict.fromkeys(clauses,0)
                        for fc in fulfilled:
                                for v in clauses[fc]:
                                        if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                                num_literals[fc] += 1
                
                return fulfilled,num_literals

        def write_literal_info(self, literal_info: dict):

                literal_info.update( {n:(v,[self.graph.nodes[n]['pos'][0],self.graph.nodes[n]['pos'][1]-0.1]) for n,v in literal_info.items()})

                for _,data in literal_info.items():
                        self.ax.text(data[1][0],data[1][1],data[0],{'color': 'black', 'ha': 'center', 'va': 'center', 'fontsize':'small'})

        def get_grasp_animation(self, i:int, log_data: list):
                if log_data[i].get('m','') in ['ch', 'li'] or log_data[i].get('status','') == 'init':
                        comment = self.get_gvns_animation(i,log_data)
                        if log_data[i].get('end_iter'):
                                j = i-1
                                while j >= 0:
                                        if log_data[j].get('end_iter') or j == 0:
                                                comment.better = log_data[i].get('best',0) > log_data[j].get('best',0)
                                                break
                                        j -= 1
                        return comment

                incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
                variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
                clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']
                data = log_data[i]
                comment_params = CommentParameters()
                comment_params.opt = Option.RGC
                comment_params.status = data.get('status','start')
                comment_params.better = data.get('better',False)
                mx = max(data.get('cl',{0:0}).values())
                mn = min(data.get('cl',{0:0}).values())
                par = data.get('par', 0)
                if type(par) == int:
                        comment_params.k = par
                else:
                        comment_params.alpha = par
                        comment_params.thres = round(mx - par * (mx-mn),2)

                if comment_params.status == 'end':
                        comment_params.best = data.get('best',0)
                        comment_params.obj = data.get('obj', 0)
                        nx.set_node_attributes(self.graph, {k: self.red if data['inc'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in incumbent}, name='color')
                        nx.set_node_attributes(self.graph, {k: self.red if data['sol'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in variables}, name='color')
                        _,_,pos_literals = self.color_and_get_changed_clauses(i,log_data)
                        self.get_flipped_variables(i,log_data)
                        self.draw_graph(incumbent if data.get('better',False) else [])
                        self.write_literal_info(pos_literals)
                        self.add_sol_description(i,data)
                        return comment_params

                nx.set_node_attributes(self.graph,{n:'' for n,t in self.graph.nodes(data='type') if t=='incumbent'}, name='label')
                j = i
                while not log_data[j].get('end_iter', False) and j > 0:
                        j = j-1
                comment_params.best = log_data[j].get('best',0)
                if comment_params.status == 'start':
                        comment_params.obj = 0
                        self.draw_graph([])
                        self.write_literal_info(dict.fromkeys(clauses,0))
                        self.add_sol_description(i,data)
                        return comment_params

                #map variable ids to node ids
                keys = {v:k for k,v in nx.get_node_attributes(self.graph,'nr').items() if self.graph.nodes[k]['type'] == 'variable'}
                rcl = [np.sign(v)*keys[abs(v)] for v in data.get('rcl',[])]
                cl = {np.sign(k)*keys[abs(k)]:v for k,v in data['cl'].items()}
                sel = keys.get(abs(data.get('sel',0)),0) * np.sign(data.get('sel',0))

                not_sel = set(abs(v) for v in cl.keys())
                selected = set(variables).difference(not_sel)
                selected.add(abs(sel))

                #set colors for edges and variables
                #set colors for clauses
                if not log_data[i-1].get('sol', False):
                        log_data[i-1]['sol'] = [-1 for _ in log_data[i]['sol']]
                added,_,pos_literals = self.color_and_get_changed_clauses(i,log_data,start=not (comment_params.status == 'sel'))
                nx.set_node_attributes(self.graph, {n: self.red if data['sol'][self.graph.nodes[n]['nr']-1] == 0 else self.blue for n in selected if n != 0}, name='color')
                self.get_flipped_variables(i,log_data)
                nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if abs(sel) in edge}, 'color')
                

                comment_params.gain = len(added)
                comment_params.obj = sum(p > 0 for p in pos_literals.values())
                # draw graph and print textual information
                self.draw_graph(([abs(sel)] if sel != 0 else []) + list(added))
                self.write_literal_info(pos_literals)
                self.write_cl_info(cl, rcl, sel)
                self.add_sol_description(i,data)
                return comment_params


        def get_ts_animation(self, i:int, log_data:list):

                if i==0 and not self.graph.ts_reposition:
                        self.reposition_variables(log_data)

                comment_params = CommentParameters()
                data = log_data[i]
                comment_params.status = data.get('status','start')

                done, lit_info = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params
                flipped_nodes = comment_params.flip 
                
                tabu_list = data.get('tabu',[])
                if not self.graph.ts_reposition:
                        for ta in tabu_list:
                                tabu_var = list(map(abs,ta[0]))
                                life = ta[1]
                                nodes = [n for n,t in self.graph.nodes(data='type') if t=='variable' and self.graph.nodes[n]['nr'] in tabu_var]
                                nx.set_node_attributes(self.graph, {n: {'tabu':True,'label':str(life)} for n in nodes})

                comment_params.ll = data.get('ll',0)
                comment_params.opt = Option.CH if data.get('m').startswith('ch') else Option.TL
                if comment_params.status == 'start':
                        flipped = [nr for n,nr in self.graph.nodes(data='nr') if n in flipped_nodes]
                        flipped = {e * (-1 if s else 1) for e,s in enumerate(data.get('sol',[]),start=1) if e in flipped}
                        comment_params.asp = any(set(t[0]) == flipped for t in tabu_list)
                elif 'ta_del' in data.keys():
                        comment_params.asp = True

                flipped_nodes = [] if comment_params.status == 'end' else comment_params.flip
                flipped_nodes += [n for n,t in self.graph.nodes(data='type') if t=='incumbent'] if data.get('better',False) else []
                self.draw_graph(flipped_nodes + list(comment_params.add.union(comment_params.remove)))
                if self.graph.ts_reposition:
                        self.draw_multi_tabu_attr_list(tabu_list)
                self.write_literal_info(lit_info)
                self.add_sol_description(i,data)
                return comment_params

        def draw_multi_tabu_attr_list(self, tabu_list):
                if len(tabu_list) == 0:
                        return
                max_size = 0.05
                size = min((0.5/(len(tabu_list)*2)), max_size)
                tabu_list.sort(reverse=True,key=lambda t:t[1])
                var_pos = {data['nr']:data['pos'] for n, data in self.graph.nodes(data=True) if data['type']=='variable'}
                y_pos = 0.3
                xmin = (var_pos[min(var_pos.keys())][0]- (-1))/(1- (-1))
                xmax = (var_pos[max(var_pos.keys())][0]- (-1))/(1- (-1))
                for ta in tabu_list:
                        self.ax.axhline(y=y_pos+size/2, xmin = xmin, xmax = xmax, color=self.grey,zorder=0)
                        for tv in ta[0]:
                                col = self.blue if tv > 0 else self.red
                                p = patches.Rectangle((var_pos[abs(tv)][0]-0.075/2, y_pos), width=0.075, height=size, ec=col, fc=col,zorder=10)
                                self.ax.add_patch(p)
                                self.ax.scatter(var_pos[abs(tv)][0],y_pos+size/2, marker='x', color='black', s=120, zorder=50)

                        self.ax.text(var_pos[min(var_pos.keys())][0]*1.2, y_pos, str(ta[1]), family='sans-serif',size='medium',weight='semibold')
                        
                        y_pos += size*2
                        
                
        def reposition_variables(self, log_data: list):
                reposition = False
                for data in log_data:
                        if any(len(ta[0]) > 1 for ta in data.get('tabu',[(set(),)])):
                                reposition = True
                                break
                if not reposition or self.graph.ts_reposition:
                        return
                nx.set_node_attributes(self.graph, {n: np.array([pos[0],0.2]) for n,pos in self.graph.nodes(data='pos') if self.graph.nodes[n]['type']=='variable'}, 'pos')
                nx.set_node_attributes(self.graph, {n: np.array([pos[0],0.9]) for n,pos in self.graph.nodes(data='pos') if self.graph.nodes[n]['type']=='incumbent'}, 'pos')
                self.graph.ts_reposition = True

        


        def write_cl_info(self, cl: dict(), rcl: list(), sel: int):

                cl_positions = {n:pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'variable'}

                col = {1:self.blue,-1:self.red,0:self.grey}

                for k,v in cl.items():
                        pos = cl_positions[abs(k)]
                        c = col[np.sign(k)] if len(rcl)==0 or k in rcl else col[0]
                        bbox = dict(boxstyle="circle",fc="white", ec=c, pad=0.2) if k == sel else None
                        self.ax.text(pos[0],pos[1]+0.2+(0.05*np.sign(k)), v, {'color': c, 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': bbox})

        def create_legend_elems(self):
                legend_elements = [
                                        Line2D([0], [0], marker='s', linestyle='none', markeredgewidth=0,
                                                markerfacecolor=self.blue, markersize=11),
                                        Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=self.grey, markersize=13), 
                                        Line2D([0], [0], marker='o', linestyle='none',
                                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11),

                                        Line2D([0], [0], marker='s', linestyle='none', markeredgewidth=0,
                                                markerfacecolor=self.red, markersize=11),
                                        Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=self.green, markersize=13),
                                        Line2D([0], [0], marker='s', linestyle='none',
                                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11),
                                ]
                description = ['','','','true/false variable','unfullfilled/fullfilled clause','change in clauses/solution']

                if self.algorithm == Algorithm.TS:
                        legend_elements.insert(3, Line2D([0], [0],linestyle='none'))
                        legend_elements.append(Line2D([0], [0], marker='X', color='w',
                                                markerfacecolor='black', markersize=13))
                        description.insert(3,'')
                        description.append('tabu attribute')

                return (tuple(legend_elements), tuple(description))


        def reset_graph(self):
                nx.set_node_attributes(self.graph, {n: self.grey if self.graph.nodes[n]['type'] != 'incumbent' else self.white for n in self.graph.nodes()}, name='color')
                nx.set_edge_attributes(self.graph, self.grey, name='color')
                nx.set_node_attributes(self.graph, 1., name='alpha')
                nx.set_node_attributes(self.graph, {n: f'x{self.graph.nodes[n]["nr"]}' for n in self.graph.nodes() if self.graph.nodes[n]['type']=='incumbent'}, name='label')
                nx.set_node_attributes(self.graph,False,name='tabu')

        def draw_graph(self, pos_change):
                self.ax.clear()
                self.ax.set_ylim(bottom=-1,top=1.0)
                self.ax.set_xlim(left=-1,right=1)
                for pos in ['right', 'top', 'bottom', 'left']: 
                        self.ax.spines[pos].set_visible(False) 

                var_inc_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t in ['variable', 'incumbent']]
                var_inc_color = [self.graph.nodes[n]['color'] for n in var_inc_nodes]
                var_inc_lcol = ['black' if self.graph.nodes[n]['type'] == 'variable' else self.yellow for n in var_inc_nodes if n in pos_change]
                var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]
                var_inc_alpha = [self.graph.nodes[n]['alpha'] for n in var_inc_nodes]

                cl_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t =='clause']
                cl_color = [self.graph.nodes[n]['color'] for n in cl_nodes]
                cl_lcol = [self.yellow for n in cl_nodes]
                cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

                edges = list(self.graph.edges())
                # draw gray and black edges seperately to avoid overpainting black edges
                e_list_gray = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] !='black']
                e_color_gray = [self.graph.edges[e]['color'] for e in e_list_gray]
                e_style_gray = [self.graph.edges[e]['style'] for e in e_list_gray]

                e_list_black = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] =='black']
                e_style_black = [self.graph.edges[e]['style'] for e in e_list_black]

                var_labels = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['type'] == 'incumbent'}

                pos = nx.get_node_attributes(self.graph, 'pos')

                nx.draw_networkx_nodes(self.graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, 
                                        alpha=var_inc_alpha, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=self.ax)
                nx.draw_networkx_nodes(self.graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=150,ax=self.ax)
                nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_gray, style=e_style_gray,ax=self.ax, edge_color=e_color_gray)
                nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_black, style=e_style_black,ax=self.ax, edge_color='black')
                nx.draw_networkx_labels(self.graph, pos, labels=var_labels,ax=self.ax)

                # drawings for tabu search
                var_labels_tabu = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['tabu']}
                tabu_nodes = list(var_labels_tabu.keys())
                x_pos = {k:[v[0]-0.04,v[1]-0.04] for k,v in pos.items()}
                nx.draw_networkx_nodes(self.graph, x_pos, nodelist=tabu_nodes,node_color='black', node_shape='X', node_size=200,ax=self.ax)
                nx.draw_networkx_labels(self.graph, pos, labels=var_labels_tabu, ax=self.ax, font_size=14, font_weight='bold', 
                                        font_color='black',horizontalalignment='left',verticalalignment='baseline')
                
        def add_sol_description(self, i, data: dict):
                inc_pos = [pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'incumbent'][0]
                curr_pos = [pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'variable'][0]
                best = data.get('status','') == 'end' or (data.get('status','') == 'start' and not data.get('m','').startswith('rgc'))
                self.ax.text(-1,inc_pos[1], 'best' if i > 0 and best else '')
                self.ax.text(-1,curr_pos[1], 'current' if i > 0 else '')






def get_visualisation(prob: Problem, alg: Algorithm, instance, log_granularity: Log, dmin: float):
    prob_class = globals()[prob.name + 'Draw']
    prob_instance = prob_class(prob, alg, instance, log_granularity, dmin)
    return prob_instance



    