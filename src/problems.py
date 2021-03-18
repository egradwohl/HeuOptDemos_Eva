"""
This module provides classes for defining optimization problems and algorithms, storing configurations of an algorithm and retrieving information for interface widgets.
"""

import sys
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from pymhlib.scheduler import Method
from pymhlib import demos
from pymhlib.settings import get_settings_parser


import enum
import os
from abc import ABC, abstractmethod
from typing import List



# path for reading instance files for runtime comparison -> pymhlib demo data
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data' + os.path.sep
# path for reading instance files for algorithm visualisation
vis_data_path = 'instances' + os.path.sep
parser = get_settings_parser()

# extend enums as needed, they hold the string values which are used for representation in widgets
class Problem(enum.Enum):
    MAXSAT = 'MAX-SAT'
    MISP = 'MAX-Independent Set'

class Algorithm(enum.Enum):
    GVNS = 'GVNS'
    GRASP = 'GRASP'
    TS = 'Tabu Search'

class Option(enum.Enum):
    CH = 'Initial Solution'
    LI = 'Local Improvement'
    SH = 'Shaking'
    RGC = 'Randomized Greedy Construction'
    TL = 'Tabu List'

class InitSolution(enum.Enum):
    random = 0
    greedy = 1

class Parameters():
    """A class for defining parameters of an option

    Attributes
        - name: name of the option which is also the name in the jupyter widget
        - callback: function callback that will be used for this option
        - param_type: type of parameter used in the callback e.g. float, int
        - value: value of parameter used in callback, if provided it represents a fixed value, otherwise it can be set in the widgets by a user
    """

    def __init__(self, name: str, callback=None, param_type: type=None, value=None):
        self.name = name
        self.callback = callback
        self.param_type = type(value) if value != None else param_type
        self.value = value

    def get_widget_info(self) -> tuple:
        """ Returns information about this parameter as (name, value)-pair, which is relevant for interface widgets"""
        return (self.name,self.param_type if self.value == None else self.value)

    def get_method(self, opt: Option, par=None) -> Method:
        """ Returns a pymhlib scheduler Method if a callback is provided, None otherwise."""
        param = par if par != None else self.value
        if self.callback == None:
            return None
        return Method(f'{opt.name.lower()}{param if param != None else ""}', self.callback, param)


class Configuration():
    """A class that holds all the configurations to start a pymhlib algorithm

    Attributes
        - name: describtion of the configuration
        - problem: type of the problem which should be solved
        - algorithm: algorithm which should be used to solve the problem
        - instance: filename of the instance that should be solved
        - options: dict of chosen options, keys must correspond to Option enums
        - runs: number of times the algorithm should be run
        - iterations: number of iterations that should be performed
        - seed: seed value that should be used for random choices
        - use_runs: True if previously saved runs should be loaded
        - saved_runs: a list of run-numbers which have been saved to a file
    """
    def __init__(self, problem: str, algorithm: str, instance: str='', options: dict=None, runs: int=1, 
                    iterations: int=50, seed: int=0, use_runs: bool=True, saved_runs: list=None, name: str=''):
        self.name = name
        self.problem = Problem(problem)
        self.algorithm = Algorithm(algorithm)
        self.instance = instance
        self.options = {} if options == None else options
        self.runs = runs
        self.iterations = iterations
        self.seed = seed
        self.use_runs = use_runs
        self.saved_runs = [] if saved_runs == None else saved_runs

    def get_inst_path(self, visualisation: bool=False) -> str:
        """ Returns the entire file path of the given instance."""
        if self.instance.startswith('random'):
            return self.instance
        if visualisation:
            return vis_data_path + self.instance
        return demo_data_path + self.instance

    def make_copy(self) -> "Configuration":
        """ Returns a deep copy of this Configuration."""
        copy = Configuration(self.problem, self.algorithm, self.instance, runs=self.runs, iterations=self.iterations, seed=self.seed, use_runs=self.use_runs, name=self.name)
        copy.options = {k: [i for i in v] if type(v) == list else v for k,v in self.options.items()}
        copy.saved_runs = [s for s in self.saved_runs]
        return copy


class ProblemDefinition(ABC):
    """A base class for problem definition to store and retrieve available algorithms, options and problem specific solution instances.

    Attributes
        - name: name of the problem, instance of Problem enum
        - options: dict of dicts of available algorithms and their corresponding available options/methodes
        - to_maximize: True, if this problem is a maximization problem, False otherwisse
    """
    def __init__(self, name: Problem, options: dict, to_maximize: bool = True):
        self.name = name
        self.options = options
        self.to_maximize = to_maximize
    

    def get_algorithms(self) -> List[str]:
        """ Returns a list of available algoriths for this problem."""
        return [k.value for k,_ in self.options.items()]

    def get_options(self, algo: Algorithm) -> dict:
        """ Returns available options for a given algorithm as dict, each entry holds a list of tuples with widget information."""
        options = {}
        for o,m in self.options[algo].items():
            options[o] = [p.get_widget_info() for p in m]
        return options

    @abstractmethod
    def get_solution(self, instance_path: str):
        pass

    def get_instances(self,visualisation) -> List[str]:
        """ Returns a list of available instance files for this problem. """
        path = vis_data_path if visualisation else demo_data_path
        if os.path.isdir(path):
            return os.listdir(path)
        return []
    

    def get_method(self, algo:Algorithm, opt: Option, name: str, par) -> Method:
        """ Creates and returns a pymhlib scheduler Method for a given method's name. """
        m = [p for p in self.options[algo][opt] if p.name == name]
        assert len(m) > 0, f'method not found: {name}'
        return m[0].get_method(opt,par)

    def get_ts_tie_options(self):

        for a in parser._actions:
            if '--mh_ts_tie_' + self.name.name.lower() in a.option_strings:
                return a.choices
        else:
            return []



class MAXSAT(ProblemDefinition):
    """
    A problem class defining the available algorithms and their available configuration options for the Max-SAT Problem.
    """

    def __init__(self):
        self.name = Problem.MAXSAT

        options = {Algorithm.GVNS: {
                                Option.CH: [Parameters(InitSolution.random.name, MAXSATSolution.construct)
                                            ,Parameters(InitSolution.greedy.name, MAXSATSolution.construct_greedy, value=InitSolution.greedy.value)
                                            ],
                                Option.LI: [Parameters('k-flip neighborhood search', MAXSATSolution.local_improve, param_type=int)],
                                Option.SH: [Parameters('k random flip', MAXSATSolution.shaking, param_type=int)]
                                },
                    Algorithm.GRASP: {
                                Option.LI: [Parameters('k-flip neighborhood search', MAXSATSolution.local_improve, param_type=int)],
                                Option.RGC: [Parameters('k-best', MAXSATSolution.greedy_randomized_construction, param_type=int),
                                                Parameters('alpha', MAXSATSolution.greedy_randomized_construction, param_type=float)]
                                },
                    Algorithm.TS: {
                                    Option.CH: [Parameters(InitSolution.random.name, MAXSATSolution.construct),
                                            Parameters(InitSolution.greedy.name, MAXSATSolution.construct_greedy, value=InitSolution.greedy.value)
                                            ],
                                Option.LI: [Parameters('k-flip neighborhood search', MAXSATSolution.local_improve_restricted, param_type=int)],
                                Option.TL: [Parameters('min length',None,int),Parameters('max length',None,int),Parameters('change (iteration)', None,int), Parameters('tie breaking',None,self.get_ts_tie_options())]
                                }
                    }

        super().__init__(Problem.MAXSAT, options)

    def get_solution(self, instance_path: str)-> MAXSATSolution:
        """ Creates a pymhlib MAXSAT instance of the given file name and returns a pymhlib MAXSATSolution object initialized with this instance. """
        instance = MAXSATInstance(instance_path)
        return MAXSATSolution(instance)

    def get_instances(self,visualisation) -> List[str]:
        """ Returns a list of available instance files for MAXSAT."""
        inst = super().get_instances(visualisation)
        return [i for i in inst if i[-3:] == 'cnf']


class MISP(ProblemDefinition):
    """
    A problem class defining the available algorithms and their available configuration options for the Max-Independent Set Problem.
    """
    def __init__(self):
        self.name = Problem.MISP
        options = {Algorithm.GVNS: {
                                Option.CH: [Parameters(InitSolution.random.name, MISPSolution.construct)
                                            ,Parameters(InitSolution.greedy.name, MISPSolution.construct_greedy, value=InitSolution.greedy.value)
                                            ],
                                Option.LI: [Parameters('two-exchange random fill neighborhood search', MISPSolution.local_improve, value=2)],
                                Option.SH: [Parameters('remove k and random fill', MISPSolution.shaking, param_type=int)]
                                }
                                ,
                    Algorithm.GRASP: {
                                Option.LI: [Parameters('two-exchange random fill neighborhood search', MISPSolution.local_improve, value=2)],
                                Option.RGC: [Parameters('k-best', MISPSolution.greedy_randomized_construction, param_type=int),
                                                Parameters('alpha', MISPSolution.greedy_randomized_construction, param_type=float)]
                              },
                    Algorithm.TS: {
                                    Option.CH: [Parameters(InitSolution.random.name, MISPSolution.construct)
                                            ,Parameters(InitSolution.greedy.name, MISPSolution.construct_greedy, value=InitSolution.greedy.value)
                                            ],
                                    Option.LI: [Parameters('two-exchange random fill neighborhood search', MISPSolution.local_improve_restricted, value=2)],
                                    Option.TL: [Parameters('min length',None,int),Parameters('max length',None,int),Parameters('change (iteration)', None,int), Parameters('tie breaking',None,self.get_ts_tie_options())]
                                
                    }
                    
                    }

        super().__init__(Problem.MISP, options)

    def get_solution(self, instance_path) -> MISPSolution:
        """ Creates a pymhlib MISP instance of the given file name and returns a pymhlib MISPSolution object initialized with this instance. """
        file_path = instance_path
        if instance_path.startswith('random'):
            file_path = "gnm" + instance_path[6:]
        instance = MISPInstance(file_path)
        return MISPSolution(instance)


    def get_instances(self, visualisation) -> List[str]:
        """ Returns a list of available instance files for MISP, for algorithm visualisation the option 'random' is added."""
        inst = super().get_instances(visualisation)
        inst = [i for i in inst if i[-3:] == 'mis']
        if visualisation:
            inst += ['random']
        return inst

