"""
Handler module which provides information about problems and algorithms for widgets in interface module. 
Issues pymhlib calls according to given configurations and returns data read from created log files.
"""

import sys
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")

import os
import logging
from typing import List, Tuple, Any
import pandas as pd


# pymhlib imports
from pymhlib.settings import settings, parse_settings, seed_random_generators
from pymhlib.log import init_logger
from pymhlib import demos
from pymhlib.gvns import GVNS
from pymhlib.ts import TS
from pymhlib.solution import Solution
from pymhlib.scheduler import Scheduler


# module imports
from .problems import Problem, Algorithm, Option, MAXSAT, MISP, Configuration, ProblemDefinition
from .logdata import read_step_log, read_sum_log, read_iter_log

# pymhlib settings
if not settings.__dict__: parse_settings(args='')
settings.mh_lfreq = 1 # log each iteration


# paths for instance files and log files
vis_instance_path = "instances" + os.path.sep
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'
step_log_path = "logs" + os.path.sep + "step.log"
iter_log_vis_path = "logs" + os.path.sep + "iter_vis.log"
sum_log_vis_path = "logs" + os.path.sep + "summary_vis.log"
iter_log_path = "logs" + os.path.sep + "iter.log"
sum_log_path = "logs" + os.path.sep + "summary.log"


# get available problems
problems = {p.name: p for p in [prob() for prob in ProblemDefinition.__subclasses__()]}

# methods used by module interface to extract information for widgets
def get_problems() -> List[str]:
    """ Loads the names of all available problems.
    
    :return: a list of available problems.
    """
    return [p.value for p in problems.keys()]

def get_instances(prob: Problem,visualisation: bool) -> List[str]:
    """ Loads the names of all available instances for the given problem.

    :param prob: Problem for which instances should be loaded
    :param visualisation: determines if the instances are loaded for step-by-step visualisation
    :return: a list of all available instances
    """
    return problems[prob].get_instances(visualisation)

def get_algorithms(prob: Problem) -> List[str]:
    """ Loads the names of all available algorithms for the given problem.

    :param prob: Problem for which algorithms should be loaded
    :return: a list of all available algorithms
    """
    return problems[prob].get_algorithms()

def get_options(prob: Problem, algo: Algorithm) -> dict:
    """ Loads the abailable options for the given problem and algorithm.

    :param prob: Problem for which options should be loaded
    :param algo: Algorithm for which options should be loaded
    :return: a dictionary of available options
    """
    return problems[prob].get_options(algo)


def run_algorithm_visualisation(config: Configuration) -> Tuple[List[dict],Any]:
    """ Runs a pymhlib algorithm according to the given configurations for step-by-step visualization.
    
    :param config: Configuration object holding all information needed for running the alogrithm
    :return: log data for visualizing the run as a list, the pymhlib problem instance that was optimized.
    """
    settings.mh_out = sum_log_vis_path
    settings.mh_log = iter_log_vis_path
    settings.mh_log_step = step_log_path 
    init_logger()

    settings.seed =  config.seed
    seed_random_generators()

    solution = run_algorithm(config,True)
    return read_step_log(config.problem.name.lower(), config.algorithm.name.lower()), solution.inst



def run_algorithm_comparison(config: Configuration) -> Tuple[pd.Series, pd.DataFrame]:
    """ Executes multiple runs of the given algorithm configuration for comparison. 
    
    :param config: Configuration object holding all information needed for running the alogrithm
    :return: pandas DataFrame of iteration data, pandas Dataframe of summary statistics.
    """
    settings.mh_out = sum_log_path
    settings.mh_log = iter_log_path
    settings.mh_log_step = 'None'
    init_logger()
    settings.seed =  config.seed
    seed_random_generators()

    for i in range(config.runs):
        _ = run_algorithm(config)
    log_df = read_iter_log(config.name)
    summary = read_sum_log()

    return log_df, summary


def run_algorithm(config: Configuration, visualisation: bool=False) -> Solution:
    """ Issues a call to pymhlib algorithm according to given configuration.

    :param config: Configuration object holding all information needed for running the alogrithm
    :param visualisation: determines if the instances are loaded for step-by-step visualisation
    :return: the optimized pymhlib Solution object.
    """
    settings.mh_titer = config.iterations

    if visualisation:
        #reset seed in case a random instance was created
        settings.seed =  config.seed
        seed_random_generators()

    # set tie breaking method (if given)
    tie_option = config.options.get(Option.TL,None)
    if tie_option != None:
        settings.__setattr__('mh_tie_breaking_'+ config.problem.name.lower(), tie_option[3][1])
    else:
        settings.__setattr__('mh_tie_breaking_'+ config.problem.name.lower(), 'random')


    # initialize solution for problem
    solution = problems[config.problem].get_solution(config.get_inst_path(visualisation))
    alg = None
    
    if config.algorithm == Algorithm.GVNS:
        alg = init_gvns(solution, config)

    if config.algorithm == Algorithm.GRASP:
        alg = init_grasp(solution, config)

    if config.algorithm == Algorithm.TS:
        alg = init_ts(solution, config)

    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()
        
    return solution


def init_gvns(solution, config: Configuration) -> Scheduler:
    """ Prepares parameters for running a GVNS.

    :param solution: pymhlib Solution object to be optimized
    :param config: Configuration object holding all information needed for running the alogrithm
    :return: pymhlib GVNS object, initialized according to given configuration
    """

    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in config.options.get(Option.CH, []) ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in config.options.get(Option.LI, []) ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in config.options.get(Option.SH, []) ]
    
    alg = GVNS(solution, ch, li, sh, consider_initial_sol=False)
    return alg


def init_grasp(solution, config: Configuration) -> Scheduler:
    """ Prepares parameters for running a GRASP.

    :param solution: pymhlib Solution object to be optimized
    :param config: Configuration object holding all information needed for running the alogrithm
    :return: pymhlib GVNS object, initialized according to given configuration to simulate a GRASP
    """
    if config.options[Option.RGC][0][0] == 'k-best':
        settings.mh_grc_par = True
        settings.mh_grc_k = config.options[Option.RGC][0][1]
    else:
        settings.mh_grc_par = False
        settings.mh_grc_alpha = config.options[Option.RGC][0][1]

    
    prob = problems[config.problem]

    ch = [prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in config.options[Option.RGC]]
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    rgc = [ prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in config.options[Option.RGC] ]

    alg = GVNS(solution,ch,li,rgc,consider_initial_sol=True)
    return alg


def init_ts(solution, config: Configuration) -> Scheduler:
    """ Prepares parameters for running a TS.

    :param solution: pymhlib Solution object to be optimized
    :param config: Configuration object holding all information needed for running the alogrithm
    :return: pymhlib TS object, initialized according to given configuration
    """
    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.TS, Option.CH, m[0], m[1]) for m in config.options[Option.CH] ]
    li = [ prob.get_method(Algorithm.TS, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    mini, maxi, change = config.options[Option.TL][0][1], config.options[Option.TL][1][1], config.options[Option.TL][2][1]
    alg = TS(solution, ch, li, mini, maxi, change)
    return alg








        















            

