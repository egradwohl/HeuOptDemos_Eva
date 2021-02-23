"""
handler module which provides information for widgets in interface module and uses widget input
to issue pymhlib calls
"""
import sys
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")

import os
import logging
import numpy as np
import pandas as pd

# pymhlib imports
from pymhlib.settings import settings, parse_settings, seed_random_generators
from pymhlib.log import init_logger
from pymhlib import demos
from pymhlib.gvns import GVNS
from pymhlib.ts import TS

# module imports
from .problems import Problem, Algorithm, Option, MAXSAT, MISP, Configuration, ProblemDefinition
from .logdata import read_step_log, read_sum_log, read_iter_log


if not settings.__dict__: parse_settings(args='')

# pymhlib settings
settings.mh_lfreq = 1


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
def get_problems():
    return [p.value for p in problems.keys()]

def get_instances(prob: Problem,visualisation):
    return problems[prob].get_instances(visualisation)

def get_algorithms(prob: Problem):
    return problems[prob].get_algorithms()

def get_options(prob: Problem, algo: Algorithm):
    return problems[prob].get_options(algo)


def run_algorithm_visualisation(config: Configuration):
    settings.mh_out = sum_log_vis_path
    settings.mh_log = iter_log_vis_path
    settings.mh_log_step = step_log_path 
    init_logger()

    settings.seed =  config.seed
    seed_random_generators()

    solution = run_algorithm(config,True)
    return read_step_log(config.problem.name.lower(), config.algorithm.name.lower()), solution.inst



def run_algorithm_comparison(config: Configuration):
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


def run_algorithm(config: Configuration, visualisation: bool=False):

    settings.mh_titer = config.iterations

    # initialize solution for problem
    solution = problems[config.problem].get_solution(config.get_inst_path(visualisation))

    if visualisation:
        #reset seed in case a random instance was created
        settings.seed =  config.seed
        seed_random_generators()

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


def init_gvns(solution, config: Configuration):

    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in config.options[Option.CH] ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in config.options[Option.SH] ]
    
    alg = GVNS(solution, ch, li, sh, consider_initial_sol=False)
    return alg


def init_grasp(solution, config: Configuration):
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


def init_ts(solution, config: Configuration):

    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.TS, Option.CH, m[0], m[1]) for m in config.options[Option.CH] ]
    li = [ prob.get_method(Algorithm.TS, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    mini, maxi, change = config.options[Option.TL][0][1], config.options[Option.TL][1][1], config.options[Option.TL][2][1]

    alg = TS(solution, ch, li, mini, maxi, change)
    return alg








        















            

