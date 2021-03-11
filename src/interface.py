"""
module which builds all necessary widgets for visualisation and runtime analysis based on information received from handler/logdata module

"""

import sys
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")

from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution

import os
import pandas as pd
import time
import matplotlib.pyplot as plt     
from IPython.display import display
import ipywidgets as widgets

from . import handler
from .problems import Problem, Algorithm, Option, InitSolution, Configuration
from . import plotting as p
from .plotting_runtime import PlotRuntime
from .logdata import Log, LogData, RunData, save_visualisation, read_from_logfile, get_log_description





def load_visualisation_settings(d_min: float = 1.14):

        interface = InterfaceVisualisation(True, d_min)
        interface.display_interface()

def load_runtime_settings():

        interface = InterfaceRuntimeAnalysis()
        interface.display_interface()



class InterfaceVisualisation():

        main_selection = widgets.Box()
        general_settings = widgets.Accordion(selected_index=None)
        problem_configuration = widgets.VBox()
        output_controls = widgets.HBox()
        run_button = widgets.Button(description='Run')
        save_button = widgets.Button(description='Save')
        output = widgets.Output()
        plot_instance = None
        log_data = None
        configuration = Configuration(Problem.MAXSAT.value, Algorithm.GVNS.value,'')

        def __init__(self, visualisation=True, dmin: float = 0.14) -> None:

                self.general_settings.children = self.init_general_settings()
                self.general_settings.set_title(0, 'General settings')
                self.problem_configuration.children = self.init_problem_configuration(visualisation)
                self.main_selection = self.init_main_selection()
                self.run_button.on_click(self.on_click_run)
                self.dmin = dmin

        def display_interface(self):
                display(self.main_selection)
                display(self.general_settings)
                display(self.problem_configuration)
                display(self.run_button, self.save_button)
                display(self.output_controls)
                display(self.output)


        def init_main_selection(self):
                main_selection = widgets.RadioButtons(options=['load from log file', 'generate new run'])
                logfile_selection = widgets.Dropdown(layout=widgets.Layout(display='None'))
                logfile_description = widgets.Output()

                def on_change_logfile_selection(change):
                        with logfile_description:
                                logfile_description.clear_output()
                                print(get_log_description(logfile_selection.value))

                def on_change_main_selection(change):
                        self.output.clear_output()
                        plt.close()
                        self.output_controls.layout.visibility = 'hidden'
                        logfile_description.clear_output()
                        if change == None or change.new == 'load from log file':
                                logfile_selection.options = os.listdir('logs' + os.path.sep + 'saved')
                                logfile_selection.value = logfile_selection.options[0]
                                self.run_button.disabled = not len(logfile_selection.options) > 0
                                self.save_button.layout.visibility = 'hidden'
                                self.problem_configuration.layout.display = self.general_settings.layout.display = 'None'
                                logfile_selection.layout.display = 'flex'
                                logfile_description.layout.display = 'flex'
                                on_change_logfile_selection(None)
                        else:
                                logfile_description.layout.display = 'None'
                                logfile_selection.layout.display = 'None'
                                self.general_settings.children = self.init_general_settings()
                                self.problem_configuration.children = self.init_problem_configuration(True)
                                self.problem_configuration.layout.display = self.general_settings.layout.display = 'flex'
                                self.run_button.disabled = False
                                self.save_button.layout.visibility = 'visible'
                                self.save_button.disabled = True
                        
                logfile_selection.observe(on_change_logfile_selection, names='value')
                main_selection.observe(on_change_main_selection, names='value')
                on_change_main_selection(None)
                return widgets.VBox([main_selection, logfile_selection, logfile_description])


        def init_general_settings(self):
                self.general_settings.selected_index = None
                seed = widgets.IntText(description='seed', value=0)
                iterations = widgets.IntText(description='iterations', value=50)
                runs = widgets.IntText(description='runs',value=1,layout=widgets.Layout(display='None'))
                use_runs = widgets.Checkbox(description='use saved runs',value=True,layout=widgets.Layout(display='None'))

                def on_change_seed(change):
                        if change.new < 0:
                                seed.value = 0
                        self.configuration.seed = seed.value
                def on_change_iterations(change):
                        if change.new < 1:
                                iterations.value = 1
                        self.configuration.iterations = iterations.value
                def on_change_runs(change):
                        if change.new < 1:
                                runs.value = 1
                        self.configuration.runs = runs.value
                def on_change_use_runs(change):
                        self.configuration.use_runs = use_runs.value

                seed.observe(on_change_seed,names='value')
                iterations.observe(on_change_iterations,names='value')
                runs.observe(on_change_runs,names='value')
                use_runs.observe(on_change_use_runs, names='value')
                return (widgets.VBox([iterations,seed,runs,use_runs]),) 


        def init_problem_configuration(self, visualisation):
                problem = widgets.Dropdown(options = handler.get_problems(), 
                                                description = 'Problem')
                instance = widgets.Dropdown(options = handler.get_instances(Problem(problem.value),visualisation),
                                                description = 'Instance')
                instance_box = widgets.HBox()

                algorithm =  widgets.Dropdown(options = handler.get_algorithms(Problem(problem.value)),
                                                description = 'Algorithm')
                options = widgets.VBox()

                def on_change_instance(change):
                        if instance.value == 'random':
                                n = widgets.IntText(value=30, description='n:',layout=widgets.Layout(width='150px'))
                                m = widgets.IntText(value=50, description='m:',layout=widgets.Layout(width='150px'))
                                def on_change_random(change):
                                        self.configuration.instance = 'random-'+str(n.value)+'-'+str(m.value)
                                n.observe(on_change_random)
                                m.observe(on_change_random)
                                instance_box.children = (instance,n,m)
                                on_change_random(None)
                        else:
                                instance_box.children = (instance,)
                                self.configuration.instance = instance.value

                def on_change_algorithm(change):
                        self.configuration.algorithm = Algorithm(algorithm.value)
                        self.configuration.options = {}
                        options.children = self.get_options(Problem(problem.value), Algorithm(algorithm.value))

                def on_change_problem(change):
                        algorithm.options = handler.get_algorithms(Problem(problem.value))
                        instance.options = handler.get_instances(Problem(problem.value),visualisation)
                        self.configuration.problem = Problem(problem.value)
                        on_change_instance(None)
                        on_change_algorithm(None)

                instance.observe(on_change_instance, names='value')
                algorithm.observe(on_change_algorithm, names='value')
                problem.observe(on_change_problem, names='value')
                on_change_problem(None)
                return (problem, instance_box, algorithm, options)


        def init_output_controls(self):
                self.output_controls.layout.visibility = 'hidden'
                play = widgets.Play(interval=1000, value=0, min=0, max=len(self.log_data.log_data)-1,
                        step=1, description="Press play")
                slider = widgets.IntSlider(value=0, min=0, max=len(self.log_data.log_data)-1,
                        step=1, orientation='horizontal',)

                def click_prev(event):
                        slider.value = slider.value - 1
                
                def click_next(event):
                        slider.value = slider.value + 1

                prev_iter = widgets.Button(description='',icon='step-backward',tooltip='previous', layout=widgets.Layout(width='50px'))
                next_iter = widgets.Button(description='',icon='step-forward', tooltip='next', layout=widgets.Layout(width='50px'))
                prev_iter.on_click(click_prev)
                next_iter.on_click(click_next)

                log_granularity = widgets.Dropdown(description='Log granularity', options=[l.value for l in Log])
                log_granularity.value = Log.StepInter.value

                def on_change_log_granularity(change):
                        next_iter = self.log_data.change_granularity(slider.value, Log(log_granularity.value))
                        self.plot_instance.log_granularity = self.log_data.current_level
                        #set max,min,value of slider and controls to appropriate iteration number
                        play.max = slider.max = len(self.log_data.log_data) - 1
                        slider.value = next_iter
                        self.animate(None)

                log_granularity.observe(on_change_log_granularity, names='value')
                widgets.jslink((play, 'value'), (slider, 'value'))

                slider.observe(self.animate, names = 'value')

                return (play, slider, prev_iter, next_iter, log_granularity)


        def on_click_run(self, event):

                log_data = list()
                if self.main_selection.children[0].value == 'load from log file':
                        log_data, instance = read_from_logfile(self.main_selection.children[1].value)
                        self.configuration.problem = Problem[log_data[0]]
                        self.configuration.algorithm = Algorithm[log_data[1]]
                        log_data = log_data[2:]
                        
                else:
                        log_data, instance = handler.run_algorithm_visualisation(self.configuration)

                self.log_data = LogData(self.configuration.problem, self.configuration.algorithm,log_data)

                # initialize graph from instance
                with self.output:
                        self.output.clear_output()
                        self.plot_instance = p.get_visualisation(self.configuration.problem, self.configuration.algorithm, instance, self.log_data.current_level, self.dmin)
                        widgets.interaction.show_inline_matplotlib_plots()


                self.output_controls.children = self.init_output_controls()
                self.output_controls.layout.visibility = 'visible'

                # copy configuration to make sure correct configuration is saved
                to_save = self.configuration.make_copy()
                def on_click_save(event):
                        save_visualisation(to_save, self.plot_instance.graph)
                        self.save_button.disabled = True

                self.save_button.on_click(on_click_save)
                self.save_button.disabled = False

                # start drawing
                self.animate(None)

        def animate(self,event):
                with self.output:
                        self.plot_instance.get_animation(self.output_controls.children[1].value, self.log_data.log_data)
                        widgets.interaction.show_inline_matplotlib_plots()

     
        def get_options(self, problem: Problem, algo: Algorithm):

                # create options widgets for selected algorithm
                if algo == Algorithm.GVNS:
                        return self.get_gvns_options(problem, algo)
                if algo == Algorithm.GRASP:
                        return self.get_grasp_options(problem, algo)
                if algo == Algorithm.TS:
                        return self.get_ts_options(problem, algo)
                return ()


        # define option widgets for each algorithm
        def get_gvns_options(self, problem: Problem, algo: Algorithm):
                options = handler.get_options(problem, algo)
                ch = widgets.Dropdown( options = [m[0] for m in options[Option.CH]],
                                description = Option.CH.value)
                ch_box = widgets.VBox([ch])
                def on_change_ch(change):
                        self.configuration.options.update( {Option.CH: [(ch.value, InitSolution[ch.value].value)]})
                on_change_ch(None)
                ch.observe(on_change_ch)

                li_box = self.get_neighborhood_options(options, Option.LI, problem, algo)
                sh_box = self.get_neighborhood_options(options, Option.SH, problem, algo)

                return (ch_box,li_box,sh_box)


        def get_grasp_options(self, problem: Problem, algo: Algorithm):
                options = handler.get_options(problem, algo)
                li_box = self.get_neighborhood_options(options, Option.LI, problem, algo)
                rcl = widgets.RadioButtons(options=[m[0] for m in options[Option.RGC]],
                                                description=Option.RGC.value)
                k_best = widgets.IntText(value=1,description='k:',layout=widgets.Layout(width='150px', display='None'))
                alpha = widgets.FloatSlider(value=0.85, description='alpha:',step=0.05,layout=widgets.Layout(display='None'), orientation='horizontal', min=0, max=1)

                rcl_box = widgets.VBox()

                def set_param(change):
                        if change['new'] == 'k-best':
                                k_best.layout.display = 'flex'
                                alpha.layout.display = 'None'
                                rcl_box.children = (rcl,k_best)
                                self.configuration.options.update( {Option.RGC: [('k-best',k_best.value)]})
                        if change['new'] == 'alpha':
                                k_best.layout.display = 'None'
                                alpha.layout.display = 'flex'
                                rcl_box.children = (rcl,alpha)
                                self.configuration.options.update( {Option.RGC: [('alpha',alpha.value)]})
                def on_change_k(change):
                        if change.new < 1:
                                change.owner.value = 1
                        self.configuration.options.update( {Option.RGC: [('k-best',k_best.value)]})
                def on_change_alpha(change):
                        self.configuration.options.update( {Option.RGC: [('alpha',alpha.value)]})

                k_best.observe(on_change_k, names='value')
                alpha.observe(on_change_alpha, names='value')
                rcl.observe(set_param, names='value')
                set_param({'new':rcl.value})

                return (li_box,rcl_box)

        def get_ts_options(self, problem: Problem, algo: Algorithm):
                options = handler.get_options(problem, algo)
                ch = widgets.Dropdown( options = [m[0] for m in options[Option.CH]],
                        description = Option.CH.value)

                ch_box = widgets.VBox([ch])
                def on_change_ch(change):
                        self.configuration.options.update( {Option.CH: [(ch.value, InitSolution[ch.value].value)]})
                on_change_ch(None)
                ch.observe(on_change_ch)
                li_box = self.get_neighborhood_options(options, Option.LI, problem, algo)
                min_ll = widgets.IntText(value=5,description='min length', layout=widgets.Layout(width='150px'), disabled=True)
                max_ll = widgets.IntText(value=5,description='max length', layout=widgets.Layout(width='150px'))
                iter_ll = widgets.IntText(value=0,description='change (iteration)', layout=widgets.Layout(width='150px'))

                def set_tl_options():
                        self.configuration.options.update( {Option.TL: [(o.description, o.value) for o in [min_ll, max_ll, iter_ll]]})

                def on_change_min(change):
                        if change.new > max_ll.value:
                                min_ll.value = max_ll.value
                        if change.new <= 0:
                                min_ll.value = 1
                        set_tl_options()
                def on_change_max(change):
                        if change.new < min_ll.value and iter_ll.value > 0:
                                max_ll.value = min_ll.value
                        if change.new <= 0:
                                max_ll.value = 1
                        if iter_ll.value == 0:
                                min_ll.value = max_ll.value
                        set_tl_options()
                def on_change_iter(change):
                        if change.new < 0:
                                iter_ll.value = 0
                        if change.new > 0:
                                min_ll.disabled = False
                        if change.new == 0:
                                min_ll.value = max_ll.value
                                min_ll.disabled = True
                        set_tl_options()

                
                min_ll.observe(on_change_min, names='value')
                max_ll.observe(on_change_max, names='value')
                iter_ll.observe(on_change_iter, names='value')
                label = widgets.Label(value='Tabu List')
                ll_box = widgets.HBox([label,min_ll,max_ll,iter_ll])
                set_tl_options()
                return (ch_box,li_box,ll_box)

        # helper functions to create widget box for li/sh neighborhoods
        def get_neighborhood_options(self, options: dict, phase: Option, problem: Problem, algo: Algorithm):
                available = widgets.Dropdown(
                                options = [m[0] for m in options[phase]],
                                description = phase.value
                )

                size = widgets.IntText(value=1,description='k: ',layout=widgets.Layout(width='150px'))
                add = widgets.Button(description='',icon='chevron-right',layout=widgets.Layout(width='60px'), tooltip='add')
                remove = widgets.Button(description='',icon='chevron-left',layout=widgets.Layout(width='60px'), tooltip='remove')
                up = widgets.Button(description='',icon='chevron-up',layout=widgets.Layout(width='30px'), tooltip='up')
                down = widgets.Button(description='',icon='chevron-down',layout=widgets.Layout(width='30px'), tooltip='down')
                
                def on_change_available(event):
                        opt = handler.get_options(problem,algo)
                        opt = [o for o in opt.get(phase) if o[0]==available.value][0]
                        size.value = 1 if opt[1] == None or type(opt[1]) == type else opt[1]
                        size.disabled = type(opt[1]) != type or opt[1] == None
                on_change_available(None)

                def on_change_size(change):
                        if change.new < 1:
                                change.owner.value = 1
                size.observe(on_change_size, names='value')

                available.observe(on_change_available, names='value')

                selected = widgets.Select(
                                options = [],
                                description = 'Selected'
                )

                def on_click_nh(event):
                        if event.tooltip == 'add':
                                selected.options += (available.value + ', k=' + str(size.value),)
                        if event.tooltip == 'remove':
                                if len(selected.options) == 0:
                                        return
                                to_remove = selected.index
                                options = list(selected.options)
                                del options[to_remove]
                                selected.options = tuple(options)
                        if event.tooltip == 'up':
                                if len(selected.options) == 0 or selected.index == 0:
                                        return
                                to_up = selected.index
                                options = list(selected.options)
                                options[to_up -1], options[to_up] = options[to_up], options[to_up-1]
                                selected.options = tuple(options)
                                selected.index = to_up -1
                        if event.tooltip == 'down':
                                if len(selected.options) == 0 or selected.index == len(selected.options) - 1:
                                        return
                                to_down = selected.index
                                options = list(selected.options)
                                options[to_down +1], options[to_down] = options[to_down], options[to_down+1]
                                selected.options = tuple(options)
                                selected.index = to_down +1

                add.on_click(on_click_nh)
                remove.on_click(on_click_nh)
                up.on_click(on_click_nh)
                down.on_click(on_click_nh)

                def on_change_selected(change):
                        self.configuration.options.update( {phase: [(name.split(',')[0], int(name.split('=')[1])) for name in list(selected.options)]} )
                selected.observe(on_change_selected)
                on_change_selected(None)
                middle = widgets.Box([size, add, remove],layout=widgets.Layout(display='flex',flex_flow='column',align_items='flex-end'))
                sort = widgets.VBox([up,down])
                
                return widgets.HBox([available,middle,selected,sort])        




class InterfaceRuntimeAnalysis(InterfaceVisualisation):


        def __init__(self, visualisation=False) -> None:
                super().__init__(visualisation)
                self.main_selection.layout.display = None
                self.reset_general_settings()
                self.problem_configuration.layout.display = 'flex'
                self.run_button.description = 'Run configuration'

                self.save_button.description = 'Save selected runs'
                self.save_button.layout = widgets.Layout(width='150px',justify_self='end')
                self.save_button.on_click(self.save_runs)

                self.reset_button = widgets.Button(description='Reset', icon='close',layout=widgets.Layout(width='100px',justify_self='end'))
                self.reset_button.on_click(self.on_click_reset)

                self.configurations = {}
                self.line_checkboxes = widgets.VBox()
                self.plot_options = self.init_plot_options()

                self.plot_instance = PlotRuntime()
                self.run_data = RunData()

        def reset_general_settings(self):
                self.general_settings.layout.display = 'flex'
                settings = self.general_settings.children[0]
                settings.layout.display = 'flex'
                settings.children[2].layout.display = settings.children[3].layout.display = 'flex'
                settings.children[0].value = 100
                settings.children[2].value = 5
                settings.children[1].value = 0
                settings.children[3].value = True

        def display_interface(self):
                display(self.general_settings)
                display(self.problem_configuration)
                display(self.run_button)
                display(self.line_checkboxes)
                display(widgets.HBox([self.save_button,self.reset_button]))
                display(self.plot_options)
                display(self.output)

        def init_plot_options(self):

                sum_radiobuttons = widgets.RadioButtons(options=[], layout=widgets.Layout(width='auto', grid_area='sum'))
                solutions = widgets.RadioButtons(options=['best solutions','current solutions'], layout=widgets.Layout(width='auto', grid_area='sol'))
                iter_slider = widgets.IntSlider(layout=widgets.Layout(padding="2em 0 0 0", grid_area='iter'), description='iteration', value=1, min=1)
                iter_slider.indent = False
                plot_options = widgets.GridBox(children=[solutions, iter_slider, sum_radiobuttons], 
                                        layout=widgets.Layout(
                                                padding='1em',
                                                border='solid black 1px',
                                                visibility='hidden',
                                                width='40%',
                                                grid_template_rows='auto auto auto auto',
                                                grid_template_columns='30% 20% 30% 20%',
                                                grid_template_areas='''
                                                "sol max median sum"
                                                "sol min mean sum"
                                                ". polygon best sum"
                                                " iter iter iter sum"
                                                '''))
                checkboxes = []

                def on_change_cb(change):
                        iter_slider.value = self.get_best_idx()
                        self.plot_comparison(iter_slider.value)

                for o in ['max','min','polygon','median','mean','best']:
                        cb =  widgets.Checkbox(description=o, layout = widgets.Layout(width='auto', grid_area=o))
                        cb.value = True if o in ['median', 'polygon'] else False
                        cb.indent= False
                        cb.observe(on_change_cb, names='value')
                        checkboxes.append(cb)


                plot_options.children += tuple(checkboxes)

                def on_change_iter(change):
                        self.plot_comparison(change.new if change.new > 0 else 1)

                iter_slider.observe(on_change_iter, names='value')

                def on_change_plotoptions(change):
                        self.plot_comparison(iter_slider.value)

                solutions.observe(on_change_plotoptions,names='value')
                sum_radiobuttons.observe(on_change_plotoptions,names='value')
                return plot_options

        

        def init_checkbox(self,name: str):
                
                def on_change(change):
                        iter_slider =  next((widget for widget in self.plot_options.children if widget.layout.grid_area == 'iter'), None)
                        iter_slider.value = self.get_best_idx()
                        self.plot_comparison(iter_slider.value)
                        
                cb = widgets.Checkbox(description=name, value=True)
                cb.observe(on_change,names='value')
                return cb


        def on_click_reset(self, event):
                self.reset_general_settings()
                self.configurations = {}
                for widget in [self.problem_configuration.children[0],self.problem_configuration.children[1],self.general_settings.children[0].children[0]]:
                        widget.disabled = False

                self.problem_configuration.children = self.init_problem_configuration(False)
                self.output.clear_output()
                plt.close()
                self.line_checkboxes.children = []
                self.plot_options.layout.visibility = 'hidden'
                self.run_data.reset()
                self.plot_instance.problem = None

        def on_click_run(self, event):
                text = widgets.Label(value='running...')
                display(text)
                # disable prob + inst + iteration + run button
                self.problem_configuration.children[0].disabled = self.problem_configuration.children[1].disabled = True
                self.general_settings.children[0].children[0].disabled = True
                self.run_button.disabled = True

                # prepare params and name, save params in dict of configurations
                params = self.prepare_and_save_configuration()

                # run algorithm with params or load data from file
                log_df,summary = self.load_datafile_or_run_algorithm(params)

                text.layout.display = 'None'
                self.run_button.disabled = False

                self.run_data.iteration_df = pd.concat([self.run_data.iteration_df,log_df], axis=1)
                self.run_data.summaries[params.name] = summary
                self.plot_instance.problem = params.problem

                # add name to checkbox list
                self.line_checkboxes.children +=(self.init_checkbox(params.name),)
                self.plot_options.layout.visibility = 'visible'

                # plot checked data
                iter_slider =  next((widget for widget in self.plot_options.children if widget.layout.grid_area == 'iter'), None)
                iter_slider.value = self.get_best_idx()
                iter_slider.max = len(self.run_data.iteration_df)

                self.plot_comparison(iter_slider.value)


        def get_best_idx(self):
                checked = [c.description for c in self.line_checkboxes.children if c.value]
                if checked == []:
                        return 1
                df = self.run_data.iteration_df[checked]
                m = 1
                if Problem(self.problem_configuration.children[0].value) in [Problem.MAXSAT,Problem.MISP]:
                        m = df.max().max()
                else:
                        m = df.min().min()
                return df.loc[df.isin([m]).any(axis=1)].index.min()

        
        def prepare_and_save_configuration(self):
                config = self.configuration.make_copy()
                name = [f'{config.algorithm.name.lower()}']
                for k,v in config.options.items():
                        if not type(k) == Option or len(v) == 0:
                                continue
                        o = k.name.lower()+ '-' + '-'.join([str(p[1]) for p in v])
                        o = o.replace('.','')
                        name += [o]
                count = len(self.line_checkboxes.children) + 1 
                config.name = str(count) + '. ' + '_'.join(name)
                self.configurations[config.name] = config
                return config

        def plot_comparison(self, i):
                stats = self.run_data.get_stat_options()
                self.plot_options.children[2].options = stats
                with self.output:
                        checked = [c.description for c in self.line_checkboxes.children if c.value]
                        if checked == []:
                                self.output.clear_output()
                                plt.close()
                                return
                        selected_iter_data = self.run_data.iteration_df[checked]
                        lines = [o.description for o in self.plot_options.children[3:] if o.value]
                        lines += ['best_sol' if self.plot_options.children[0].value.startswith('best') else 'current_sol']
                        sum_option = self.plot_options.children[2].value
                        selected_sum_data = {name:df[sum_option] for name,df in self.run_data.summaries.items() if name in checked}
                        self.plot_instance.plot(i, lines, self.configurations, selected_iter_data, selected_sum_data)

                        widgets.interaction.show_inline_matplotlib_plots()


        def load_datafile_or_run_algorithm(self,params: Configuration):

                if params.use_runs:
                        name = f'i{params.iterations}_s{params.seed}_' + params.name[params.name.find('.')+1:].strip()
                        description = self.create_configuration_description(params)
                        file_name = self.configuration_exists_in_saved(name,description)
                        if file_name:
                                f = open(file_name, 'r')
                                ex_runs = int(f.readline().split('=')[1].strip())
                                if params.runs <= ex_runs:
                                        data, sm = self.run_data.load_datafile(file_name,params.runs)
                                        data.columns = pd.MultiIndex.from_tuples(zip([params.name]*len(data.columns), data.columns))
                                        self.configurations[params.name].saved_runs = list(data.columns.get_level_values(1))
                                        return data, sm
                                if params.seed == 0:
                                        runs = params.runs
                                        # load existing runs
                                        data, sm = self.run_data.load_datafile(file_name,ex_runs)
                                        data.columns = pd.MultiIndex.from_tuples(zip([params.name]*len(data.columns), data.columns))
                                        # generate runs-ex_runs new ones and set correct run numbers
                                        params.runs = runs-ex_runs
                                        new_data, new_sm = handler.run_algorithm_comparison(params)
                                        params.runs = runs
                                        new_data.columns = pd.MultiIndex.from_tuples([(n,int(r)+ex_runs) for (n,r) in new_data.columns])
                                        new_sm.index = pd.MultiIndex.from_tuples([(int(r)+ex_runs,m) for (r,m) in new_sm.index])

                                        self.configurations[params.name].saved_runs = list(data.columns.get_level_values(1))
                                        # concatenate them
                                        return pd.concat([data,new_data],axis=1),pd.concat([sm,new_sm])
                                
                return handler.run_algorithm_comparison(params)


        def save_runs(self,event):

                checked = {config.description for config in self.line_checkboxes.children if config.value}
                not_saved = {name for name,config in self.configurations.items() if config.runs > len(config.saved_runs)}
                to_save = checked.intersection(not_saved)

                for s in to_save:
                        config = self.configurations[s]
                        description = self.create_configuration_description(config)
                        name = f'i{config.iterations}_s{config.seed}_' + s[s.find('.')+1:].strip()
                        filepath = self.configuration_exists_in_saved(name, description)

                        if filepath:
                                if config.seed == 0:
                                        self.run_data.save_to_logfile(config,filepath,append=True)
                                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))
                                        return
                                elif config.runs <= len(config.saved_runs):
                                        # do nothing, only existing runs were loaded
                                        return
                                else:
                                        # overwrite existing file
                                        self.run_data.save_to_logfile(config,filepath)
                                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))
                                        return
                        # create new file and write to it
                        filepath = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + config.problem.name.lower() + os.path.sep +\
                                          f'r{config.runs}_' + name + '_' +config.instance+ '_'+ time.strftime('_%Y%m%d_%H%M%S') + '.log'
                        self.run_data.save_to_logfile(config,filepath,description=description)
                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))


        def configuration_exists_in_saved(self, name: str, description: str):
                description = description[description.find('D '):]
                path = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + Problem(self.problem_configuration.children[0].value).name.lower()
                log_files = os.listdir(path)
                for l in log_files:
                        n = l[l.find('i'):]
                        if n.startswith(name):
                                file_des = ''
                                with open(path + os.path.sep + l) as file:
                                        for line in file:
                                                if line.startswith('R'):
                                                        continue
                                                if line.startswith('D'):
                                                        file_des += line
                                                else:
                                                        break
                                if file_des.strip() == description:
                                        return path + os.path.sep + l
                return False
        

        def create_configuration_description(self, config: Configuration):
                s = f"R runs={config.runs}\n"
                s += f'D inst={config.instance}\n'
                s += f"D seed={config.seed}\n"
                s += f"D iterations={config.iterations}\n"

                for o,v in config.options.items():
                        v = v if type(v) == list else [v]
                        for i in v:
                                s += f'D {o.name}{i}\n'

                return s.strip()
