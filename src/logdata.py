""" Module for reading log files created by pymhlib runs and preparing the data for visualisation.
Classes for handling visualisation data.
"""

import sys
from typing import List
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")
from pymhlib.demos.misp import MISPInstance
from pymhlib.demos.maxsat import MAXSATInstance


import ast
import re
import os
import numpy as np
import enum
import time
import pandas as pd

from .problems import Configuration, Problem, Algorithm, Option


class Log(enum.Enum):
    """ Enumerations for different levels of log granularity for step by step visualization.
    The values of the attributes correspond to their names in the widgets.
    """
    StepInter = 'step-by-step (intermediate steps)' # start-frame and end-frame for each step
    StepNoInter = 'step-by-step (no intermediate steps)' # start and end combined in one frame
    NewInc = 'new incumbents' # only frames where new best solution was found
    Update = 'updated solutions' # result of a phase e.g. li(vnd)-cycle, complete rgc in one frame
    Cycle = 'major cycles' # result of one entire cycle of an algorithm, e.g. sh+li (gvns), rgc+li (grasp), per frame


# global variables
len_start = 7 # default number of lines that hold all information logged before an algorithm method is performed
len_end = 8 # default number of lines that hold all information logged after an algorithm method was performed
step_log_path = 'logs'+ os.path.sep + 'step.log'
iter_log_path = "logs" + os.path.sep + "iter.log"
sum_log_path = "logs" + os.path.sep + "summary.log"


class LogData():
    """A class for storing log data for step-by-step visualization. 
    Provides methods and attributes for changing the level of log granularity and retrieving the relevant data.

    Attributes
        - problem: Problem used for this log data
        - algorithm: Algorithm used for this log data
        - full_data: list of dictionaries, holds the entire visualization data
        - levels: dictionary keyed by levels of log granularity, each holds a list of indices of the relevant data
        - current_level: holds the currently active level of log granularity
        - log_data: list of dicionaries, holds the data for the currently active level of log granularity
    """

    def __init__(self, problem: Problem, algorithm: Algorithm, log_data: list):
        """ Initialization.

        :param problem: Problem used for the given log data
        :param algorithm: Algorithm used for the given log data
        :param log_data: list holding the entire data read from the log file
        """
        self.problem = problem
        self.algorithm = algorithm
        self.full_data = [{'status':'init', 'algorithm': self.algorithm, 'problem': self.problem}] + log_data #prepend additional frame for displaying instance in the beginning
        self.levels = self.init_levels()
        self.current_level = Log.StepInter
        self.log_data = self.full_data # holds logdata for currently active log level

    def init_levels(self) -> dict:
        """ Dedects the idices of the relevant data for each level of log granularity out of the complete log data.
        
        :return: a dictionary keyed by levels of log granularity, each holding a list of corresponding indices
        """
        levels = dict()
        levels[Log.StepInter] = list(range(len(self.full_data)))
        levels[Log.StepNoInter] = [0] +[i for i in levels[Log.StepInter] if self.full_data[i].get('status', '') in  ['end', 'sel']]
        levels[Log.NewInc] =[0] + [i for i in levels[Log.StepNoInter] if self.full_data[i].get('better',False)]
        update = list()
        if self.algorithm == Algorithm.TS:
            update = [i for i in levels[Log.StepNoInter]]
        if self.algorithm == Algorithm.GRASP or self.algorithm == Algorithm.GVNS:
            for i,j in enumerate(levels[Log.StepNoInter][:-1]):
                if self.full_data[j].get('m','') != self.full_data[levels[Log.StepNoInter][i+1]].get('m',''):
                    update.append(j)
            update.append(levels[Log.StepNoInter][-1])

        levels[Log.Update] = update
        levels[Log.Cycle] =[ i for i in levels[Log.Update] if self.full_data[i].get('m','') in ['ch', 'li'] or i== 0]
        if self.full_data[levels[Log.Update][-1]].get('m','') != 'li' and not levels[Log.Update][-1] in levels[Log.Cycle]:
            levels[Log.Cycle].append(levels[Log.Update][-1])
        for i in levels[Log.Cycle][1:]:
            self.full_data[i]['end_iter'] = True

        return levels


    def change_granularity(self, i: int, granularity: Log) -> int:
        """ Changes the currently active log granularity and determines the index of the next data that should be displayed for the new log granularity.

        :param i: number of iteration at the time of changing the log level
        :param granularity: new level of log granularity
        :return: the index of the next data that should be displayed for the new log granularity
        """
        self.log_data = [self.full_data[i] for i in self.levels[granularity]]
        current_iter = self.levels[self.current_level][i]
        next_iter = len(self.levels[granularity]) -1  if current_iter > self.levels[granularity][-1] else next(i for i,val in enumerate(self.levels[granularity]) if val >= current_iter) 
        self.current_level = granularity
        return next_iter



def read_step_log(prob: str, alg: str):
    """Method for reading data logged by pymhlibs step-logger and creating data dicts for algorithm visualisation.

    :return: a list of dicts, each containing the data used for plotting one visualisation frame.
    """
    data = list()
    with open(step_log_path, 'r') as logfile:
        for line in logfile:
            data.append(cast_line(line.strip()))
        logfile.close()

    return create_log_data(data)



def read_sum_log():
    """Reads algorithm summaries of n runs logged by pymhlibs general logger.

    :return: a dataframe containing algorithm statistics for n runs.
    """
    idx = []
    with open(sum_log_path) as f: 
        for i, line in enumerate(f):
            if not line.startswith('S '):
                idx.append(i)
        f.close()
        
    df = pd.read_csv(sum_log_path, sep=r'\s+',skiprows=idx)
    df.drop(labels=['S'], axis=1,inplace=True)
    idx = df[ df['method'] == 'method' ].index
    df.drop(idx , inplace=True)

    n = len(df[df['method'] == 'SUM/AVG'])
    m = int(len(df) / n)
    df['run'] = (np.array([i for i in range(1,n+1)]).repeat(m))
    df.set_index(['run','method'], inplace=True)
    return df


def read_iter_log(name):
    """Reads iteration data logged by pymhlibs iteration logger.

    :param name: name of the configuration the data belongs to.
    :return: a dataframe containing objective values of iterations for n runs.
    """
    df = pd.read_csv(iter_log_path, sep=r'\s+', header=None)

    df.drop(df[ df[1] == '0' ].index , inplace=True) #drop initialisation line
    df = df[4].reset_index().drop(columns='index') #extract 'obj_new'
    indices = list((df[df[4] == 'obj_new'].index)) + [len(df)] #get indices of start of each run
    list_df = []
    #split data in single dataframes
    for i in range(len(indices) - 1):
        j = indices[i+1]-1
        frame = df.loc[indices[i]:j]
        frame = frame.reset_index().drop(columns='index')
        list_df.append(frame)
    full = pd.concat(list_df,axis=1) #concatenate dataframes
    full.columns = [i for i in range(1,len(full.columns)+1)] #rename columns to run numbers
    full.columns = pd.MultiIndex.from_tuples(zip([name]*len(full.columns), full.columns)) # set level of column
    full = full.drop([0]) #drop line that holds old column names
    full = full.astype(float)
    
    return full


def create_log_data(data: list) -> List[dict]:
    """ Processes the raw data read from the log file, converts it into a list where each entry represents the data for one
    visualization frame.

    :param data: raw data, each entry corresponds to a line of the log file
    :return: data for visualization
    """

    vis_data = []
    i = 0
    while i < len(data):

        start = i
        while not data[start] == 'start':
            start += 1
        end = start
        while not data[end] == 'end':
            end +=1
        len_end = end
        while not data[len_end] == 'start':
            len_end += 1
            if len_end >= len(data):
                break

        method = data[start+3]['m']

        # create data from start-end-slices according to method
        if method in ['ch','li', 'sh']:
            vis_data.append(create_gvns_data(data[start:end]))
            vis_data.append(create_gvns_data(data[end:len_end]))
        if method in ['rgc']:
            vis_data += create_grasp_data(data[start:len_end])

        i = len_end

    return vis_data


def create_gvns_data(data: list) -> dict:
    """ Creates visualisation data for one gvns frame out of a list of log file data.
    Is also used for data of a tabu search.
    
    :param data: a list of one-entry dictionaries that describe one visualisation frame
    :return: a dictionary holding the data for one gvns/ts frame
    """
    entries = {k:v for x in data if type(x) == dict for k,v in x.items() if k!='ta'}
    tabu_attr = [v for x in data if type(x) == dict for k,v in x.items() if k=='ta']
    if len(tabu_attr) > 0 :
        entries['tabu'] = tabu_attr
    entries['status'] = data[0]
    return entries


def create_grasp_data(data: list) -> dict:
    """ Creates visualisation data for one grasp frame out of a list of log file data.
    
    :param data: a list of one-entry dictionaries that describe one visualisation frame
    :return: a dictionary holding the data for one grasp frame
    """
    entries = [create_gvns_data(data[:len_start])]
    end_i = data.index('end')

    greedy_data = data[len_start:end_i]

    for i in range(0, len(greedy_data),5):
        rcl_data = {list(d.keys())[0]:list(d.values())[0] for d in greedy_data[i:i+5]}

        entries.append({'m':'rgc', 'status':'cl', 'cl':rcl_data['cl'], 'sol':rcl_data['sol'], 'par':rcl_data['par']})
        entries.append({'m':'rgc','status':'rcl', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':rcl_data['sol'], 'par':rcl_data['par']})
        sol =  data[end_i+1]['sol']  if i == len(greedy_data) -5 else greedy_data[i+5]['sol']
        entries.append({'m':'rgc','status':'sel', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':sol, 'sel':rcl_data['sel'], 'par':rcl_data['par']})

    entries.append(create_gvns_data(data[end_i:]))
    return entries

        
def cast_line(line: str):
    """ Casts the given string to the appropriate data type.
    
    :param line: one line of the log file
    :return: if the string is of the form 'key: data', a dictionary is returned, else the input line is returned
    """
    if not ':' in line:
        return line.lower()
    idx = line.find(':')
    name, data = line[:idx].strip().lower(), line[idx+1:].strip()

    if re.match("^[+-]?[0-9]+(\.)?[0-9]*", data): #numerical data
        return {name: cast_number(data)}

    if re.match("^[a-z]+[0-9]+(\.)?[0-9]*", data): # method with parameter
        #extract method name
        x = re.search("^[a-z]+", data)
        return {name: x.group() }

    if data in ['False', 'True']:
        return {name: False if data=='False' else True}

    x = re.search(r'(?<=\()(.*?)(?=\))', data)
    if x: #tuple
        y = '(' + x.group()+')'
        return {name: ast.literal_eval(y)}

    x = re.search(r'(?<=\[)(.*?)(?=\])', data)
    if x: #string representation of numpy array
        x = x.group()
        x = ' '.join(x.split())
        x = "[" + x.replace(" ",",") + "]"
        return {name: ast.literal_eval(x)} 

    x = re.search(r'(?<=\{)(.*?)(?=\})', data)
    if x: #dict
        x = "{" + x.group() + "}"
        return {name: ast.literal_eval(x)}

    return {name: data}


def cast_number(data: str):
    """ Casts the given numeric string to a number.

    :param data: numeric string
    :return: a number (int or float)
    """

    if re.match("^[-+]?[0-9]+$", data): #int
        return int(data)

    if re.match("^[-+]?[0-9]+\.[0-9]*", data): #float
        return float(data)

    return data


def save_visualisation(params: Configuration, graph=None):
    """ Saves the current content of the visualisation log file written by the pymhlib step logger to a dedicated directory.
    Prepends textual information about the used configuration. If a randomly created instance was used, it is saved to 
    an instance folder for later use (currently only available for MISP).

    :param params: configuration that was used for running the pymhlib algorithm
    :param graph: a networkx graph instance that was used as input for the pymhlib algorithm
    """
    # if instance==random, create instance file from graph, save in instance folder and keep filename
    inst_filename = params.instance
    if inst_filename.startswith('random'):
        # for now only available for misp
        if params.problem.name == 'MISP':
            inst_filename = save_misp_instance(graph)

    # get current log file according to problem and algo and copy content
    logfile = 'logs' + os.path.sep + 'step.log'
    with open(logfile, 'r') as source:
        timestamp = time.strftime('_%Y%m%d_%H%M%S')
        with open(os.path.sep.join(['logs','saved',params.problem.name.lower()+ '_' + params.algorithm.name.lower() + timestamp + '.log']), 'w') as destination:
            data = source.read()
            # prepend description block to log file (instance filename, options)
            destination.write(f'P: {params.problem.name}\nA: {params.algorithm.name}\n')
            destination.write('I: ' + inst_filename + '\n')
            for k,v in params.options.items():
                if type(k) == Option:
                    destination.writelines( [f'O: {k.name} {o}\n' for o in v] )
            destination.write(data)
            source.close()
            destination.close()


def save_misp_instance(graph):
    """ Saves the given networkx graph as instance file to be used as input for MISPs.

    :param graph: a networkx graph instance
    :return: the name of the file the instance was saved to
    """
    filename = '_'.join(['gnm', str(graph.order()), str(graph.size()), time.strftime('%Y%m%d%H%M%S')]) + '.mis'
    pathname = 'instances' + os.path.sep + filename
    with open(pathname, 'w') as inst_file:
        inst_file.writelines(['c "source: networkx.gnm_random_graph()"\n', f'p edge {graph.order()} {graph.size()}\n'])
        for u,v in graph.edges():
            inst_file.write(f'e {u+1} {v+1}\n')
        inst_file.close()
    return filename


def read_from_logfile(filename: str):
    """ Reads log data from the given file name, prepares the data for step-by-step visualisation and 
    creates an Instance object from the information in the log file.

    :param filename: name of the log file to be read
    :return: a list holding the visualisation data and the problem instance object
    """
    data = list()
    instance_file = ''
    probl = ''
    algo = ''
    with open('logs' + os.path.sep + 'saved' + os.path.sep + filename, 'r') as logfile:
        for line in logfile:
            if line.startswith('I:'):
                instance_file = line.split(':')[1].strip()
                continue
            if line.startswith('O:'):
                continue
            if line.startswith('P:'):
                probl = line.split(':')[1].strip()
                continue
            if line.startswith('A:'):
                algo = line.split(':')[1].strip()
            data.append(cast_line(line.strip()))
        logfile.close()
    
    instance_path = 'instances' + os.path.sep + instance_file
    inst = None
    if probl == Problem.MISP.name:
        inst = MISPInstance(instance_path)
    if probl == Problem.MAXSAT.name:
        inst = MAXSATInstance(instance_path)
    vis_data = create_log_data(data)
    vis_data = [probl] + [algo] + vis_data
    return vis_data, inst

def get_log_description(filename: str):
    """ Read the textual description from the beginning of the given file.

    :param filename: name of the file to be read
    :return: desription of the log file as multi-line string
    """
    if not filename:
        return ''
    description = []
    with open('logs' + os.path.sep + 'saved' + os.path.sep + filename, 'r') as logfile:
        for line in logfile:
            if line.startswith('I:'):
                description.append('Instance: ' + line[2:].strip())
            elif line.startswith('O:') or line.startswith('P:') or line.startswith('A:'):
                description.append(line[2:].strip())
            else:
                break
    
    return '\n'.join(description)


class RunData():
    """ Class for storing data of multiple algorithm runs of different configurations.
    Provides methods for plotting the selected data.

    Attributes:
    - summaries: dictionary, keys=names of the configurations, values=pandas data frames of the summary statistics
    - iteration_df: pandas data frame of the iteration data (objective values) of all runs currently available
    """

    def __init__(self):
        """ Initialization.
        """
        self.summaries = dict()
        self.iteration_df = pd.DataFrame()

    def reset(self):
        """ Removes currently stored data.
        """
        self.summaries = dict()
        self.iteration_df = pd.DataFrame()

    def get_stat_options(self):
        """ Reads available summary statistic categories from the summaries data frame.

        :return: tuple of available summary statistics
        """
        if len(self.summaries) == 0:
            return tuple()
        return tuple(list(self.summaries.values())[0].columns)

    def save_to_logfile(self, config: Configuration, filepath: str, description: str=None, append: bool=False):
        """ Saves all runs of the given configuration that have not yet been saved to a data file.
        If runs are appended to a file, the number of runs in the file description and the file name is increased accordingly.

        :param config: configuration containing information about what runs should be saved.
        :param filepath: file name the runs should be saved to
        :param description: if the file already exists, the parameter holds the description of the file, else its None
        :param append: True if the runs are to be appended to a file, False if any existing file is to be overwritten
        """
        mode = 'w' if description else 'r+'
        f = open(filepath, mode)

        if description:
                f.write(description+'\n')
                df = self.iteration_df[config.name].T
                df.to_csv(f,sep=' ',na_rep='NaN', mode='a', line_terminator='\n')
                f.write('S summary\n')
                self.summaries[config.name].to_csv(f,na_rep='NaN', sep=' ',mode='a',line_terminator='\n')
                f.close()
        else:
                saved_runs = set(config.saved_runs)
                runs = set(range(1,config.runs+1))
                to_save = list(runs - saved_runs)
                to_save.sort()
                if len(to_save) == 0:
                        f.close()
                        return
                data = f.readlines()
                df = self.iteration_df[config.name][to_save].T
                sm = self.summaries[config.name].loc[to_save]
                existing_runs =int(data[0].split('=')[1].strip())
                if append: #seed==0

                        idx = next((i for i,v in enumerate(data) if v.startswith('S ')), 0)
                        df.index = pd.Index(range(existing_runs+1,existing_runs+1+len(to_save)))
                        sm.index = pd.MultiIndex.from_tuples(zip(df.index.repeat(len(sm)/len(to_save)),sm.index.get_level_values(1)),names=sm.index.names)
                        sm.reset_index(inplace=True)
                        data.insert(idx, df.to_csv(sep=' ',line_terminator='\n',header=False))
                        data += [sm.to_csv(sep=' ',line_terminator='\n', index=False,header=False)]
                        data[0] = f'R runs={df.index[-1]}\n'
                        f.seek(0)
                        f.writelines(data)
                        f.truncate()
                        f.close()
                        
                        os.rename(filepath,filepath.replace(f'r{existing_runs}',f'r{existing_runs+len(to_save)}',1))
                    
                else: # seed!= 0
                        if len(runs) <= existing_runs:
                                f.close()
                                return
                        data[0] = f"R runs={config.runs}\n"
                        idx = next((i for i,v in enumerate(data) if not v[0] in ['R','D']), 0)
                        data = data[:idx]
                        data += [df.to_csv(sep=' ',line_terminator='\n')]
                        data += ['S summary\n']
                        data += [sm.to_csv(sep=' ',line_terminator='\n')]
                        f.seek(0)
                        f.writelines(data)
                        f.truncate()
                        f.close()

                        os.rename(filepath,filepath.replace(f'r{existing_runs}',f'r{len(to_save)}',1))


    def load_datafile(self,filename,runs: int):
        """ Loads the given number of runs from the given file.

        :param filename: name of the file from which runs are to be loaded
        :param runs: number of runs that are to be loaded
        :return: a tuple holding iteration data (objective values) and summary statistics of the requested runs
        """
        f = open(filename, 'r')
        pos = 0
        while True:
                l = f.readline()
                if not l[0] in ['D','R']:
                        break
                pos = f.tell()
        f.seek(pos)
        data = pd.read_csv(f, sep=r'\s+', nrows=runs).T
        data.reset_index(drop=True, inplace=True)
        data.index += 1
        f.seek(pos)
        
        while True:
                l = f.readline()
                if l[0] == 'S':
                        break
        sm = pd.read_csv(f,sep=r'\s+',index_col=['run','method'])
        sm = sm[sm.index.get_level_values('run') <= runs]
        f.close()

        return data,sm     
      

