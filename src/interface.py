"""
module which builds all necessary widgets for visualisation and runtime analysis (TODO) based on information of handler module

"""
import networkx as nx
import matplotlib.pyplot as plt
import statistics        
from IPython.display import display, display_html 
import ipywidgets as widgets
import src.handler as handler
from src.handler import Problem, Algorithm, Option
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
import src.plotting as p
from src.logdata import Log, LogData, save_visualisation, read_from_logfile, get_log_description
from IPython.display import clear_output
import os
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib import gridspec as gs
import time




def load_visualisation_settings():

        interface = InterfaceVisualisation()
        interface.display_main_selection()

def load_runtime_settings():

        interface = InterfaceRuntimeAnalysis()
        interface.display_widgets()



class InterfaceVisualisation():

        def __init__(self,visualisation=True):

                self.problemWidget = widgets.Dropdown(
                                        options = handler.get_problems(),
                                        description = 'Problem'
                                        )
                self.algoWidget =  widgets.Dropdown(
                                options = handler.get_algorithms(Problem(self.problemWidget.value)),
                                description = 'Algorithm'
                                )
                
                self.instanceWidget = widgets.Dropdown(
                                options = handler.get_instances(Problem(self.problemWidget.value),visualisation),
                                description = 'Instance'
                                )
                self.instanceBox = widgets.HBox([self.instanceWidget])

                self.optionsWidget = widgets.VBox() #container which holds all options

                self.optionsHandles = {} #used to store references to relevant option widgets since they are wrapped in boxes

                self.log_data = None

                self.plot_instance = None
                self.out = widgets.Output()
                self.settingsWidget = widgets.Accordion(selected_index=None)
                iterations = widgets.IntText(description='iterations', value=100)
                seed = widgets.IntText(description='seed', value=0)
                self.settingsWidget.children = (widgets.VBox([iterations,seed]),)
                self.settingsWidget.set_title(0, 'General settings')
                self.controls = self.init_controls()

                self.controls.layout.visibility = 'hidden'
                self.run_button =  widgets.Button(description = 'Run')
                self.save_button = None
                self.mainSelection = widgets.RadioButtons(options=['load from log file', 'generate new run'])
                self.logfileWidget = widgets.Dropdown(layout=widgets.Layout(display='None'))


        def init_controls(self):
        
                play = widgets.Play(interval=1000, value=0, min=0, max=100,
                        step=1, description="Press play")
                slider = widgets.IntSlider(value=0, min=0, max=100,
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
                log_granularity.observe(self.on_change_log_granularity, names='value')
                widgets.jslink((play, 'value'), (slider, 'value'))
                slider.observe(self.animate, names = 'value')

                return widgets.HBox([play, slider, prev_iter, next_iter, log_granularity])

        def on_change_log_granularity(self, change):
                next_iter = self.log_data.change_granularity(self.controls.children[1].value, Log(self.controls.children[4].value))
                #set max,min,value of slider and controls to appropriate iteration number
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) - 3
                self.controls.children[1].value = next_iter
                self.animate(None)



        def animate(self,event):
                with self.out:
                        #p.get_animation(self.controls.children[1].value, self.log_data.log_data, self.plot_instance)
                        self.plot_instance.get_animation(self.controls.children[1].value, self.log_data.log_data)
                        widgets.interaction.show_inline_matplotlib_plots()
                
        def on_change_problem(self, change):

                self.algoWidget.options = handler.get_algorithms(Problem(change.new))
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.instanceWidget.options = handler.get_instances(Problem(change.new), not isinstance(self, InterfaceRuntimeAnalysis))
                self.optionsHandles = {}
                self.on_change_algo(None)


    
        def on_change_algo(self, change):

                self.optionsHandles = {} #reset references
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))


        def run_visualisation(self, event):
                params = dict()
                log_data = list()
                if self.mainSelection.value == 'load from log file':
                        log_data, instance = read_from_logfile(self.logfileWidget.value)
                        params.update({'prob':Problem[log_data[0].upper()], 'algo':Algorithm[log_data[1].upper()]})
                        
                else:
                        params = self.prepare_parameters()
                        # starts call to pymhlib in handler module
                        log_data, instance = handler.run_algorithm_visualisation(params)

                self.log_data = LogData(log_data)
                #print('\n'.join([str(l) for l in self.log_data.log_data]))

                # initialize graph from instance
                with self.out:
                        self.out.clear_output()
                        self.plot_instance = p.get_visualisation(params['prob'],params['algo'], instance)
                        widgets.interaction.show_inline_matplotlib_plots()

                self.controls.children[1].value = 0
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) - 3
                self.controls.children[4].value = Log.StepInter.value

                # start drawing
                self.animate(None)
                self.controls.layout.visibility = 'visible'
                self.save_button.disabled = self.mainSelection.value == 'load from log file' 

        def prepare_parameters(self):
                # prepare current widget parameters for call to run algorithm
                params = {'prob':Problem(self.problemWidget.value),
                                'algo':Algorithm(self.algoWidget.value),
                                'inst':'-'.join([str(c.value) for c in self.instanceBox.children])}

                # store each option as list of tuples (<name>,<parameter>)
                # extend if further options are needed
                if Option.CH in self.optionsHandles:
                        params[Option.CH] = [(self.optionsHandles.get(Option.CH).value, 0)]
                if Option.LI in self.optionsHandles:
                        #TODO: make sure name splitting works if no 'k=' given (ok for now because k is always added)
                        params[Option.LI] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.LI).options)]
                if Option.SH in self.optionsHandles:
                        params[Option.SH] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.SH).options)]
                if Option.RGC in self.optionsHandles:
                        params[Option.RGC] = [(self.optionsHandles[Option.RGC].children[0].value,self.optionsHandles[Option.RGC].children[1].value)]

                # add settings params
                settings = {c.description:c.value for c in self.settingsWidget.children[0].children}
                params['settings'] = settings
                return params


        def display_main_selection(self):

                options_box = self.display_widgets()
                options_box.layout.display = 'None'
                log_description = widgets.Output()

                def on_change_logfile(change):
                        with log_description:
                                log_description.clear_output()
                                print(get_log_description(change['new']))

                self.logfileWidget.observe(on_change_logfile, names='value')

                def on_change_main(change):
                        if change['new'] == 'load from log file':
                                self.logfileWidget.options = os.listdir('logs' + os.path.sep + 'saved')
                                self.run_button.disabled = not len(self.logfileWidget.options) > 0
                                self.logfileWidget.layout.display = 'flex'
                                log_description.layout.display = 'flex'
                                options_box.layout.display = 'None'

                        else: 
                                self.run_button.disabled = False
                                self.logfileWidget.layout.display = 'None'
                                log_description.layout.display = 'None'
                                options_box.layout.display = 'flex'


                self.mainSelection.observe(on_change_main, names='value')
                self.run_button.on_click(self.run_visualisation)
                display(self.mainSelection)
                on_change_main({'new': 'load from log file'})
                display(widgets.VBox([self.logfileWidget,log_description]))
                display(options_box)
                display(widgets.VBox([self.run_button, self.controls, self.out]))


        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.instanceWidget.observe(self.on_change_instance, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))

                self.save_button = widgets.Button(description='Save Visualisation', disabled=True)
                self.save_button.on_click(self.on_click_save)
                optionsBox = widgets.VBox([self.settingsWidget, self.problemWidget,self.instanceBox,self.algoWidget,self.optionsWidget])
                return widgets.VBox([optionsBox, self.save_button])

        def on_click_save(self,event):
                #TODO make sure params were not changed!!!!
                save_visualisation(self.prepare_parameters(), self.plot_instance.graph)
                self.save_button.disabled = True


        def on_change_instance(self,change):
                if change.new == 'random':
                        n = widgets.IntText(value=30, description='n:',layout=widgets.Layout(width='150px'))
                        m = widgets.IntText(value=50, description='m:',layout=widgets.Layout(width='150px'))
                        self.instanceBox.children = (self.instanceWidget,n,m)
                else:
                        self.instanceBox.children = (self.instanceWidget,)

                
        def get_options(self, algo: Algorithm):

                options = handler.get_options(Problem(self.problemWidget.value), algo)

                # create options widgets for selected algorithm
                if algo == Algorithm.GVNS:
                        return self.get_gvns_options(options)
                if algo == Algorithm.GRASP:
                        return self.get_grasp_options(options)
                return ()


        # define option widgets for each algorithm
        def get_gvns_options(self, options: dict):

                ch = widgets.Dropdown( options = [m[0] for m in options[Option.CH]],
                                description = Option.CH.value)
                ch_box = widgets.VBox([ch])
                self.optionsHandles[Option.CH] = ch

                li_box = self.get_neighborhood_options(options, Option.LI)
                sh_box = self.get_neighborhood_options(options, Option.SH)

                return (ch_box,li_box,sh_box)


        def get_grasp_options(self, options: dict):

                li_box = self.get_neighborhood_options(options, Option.LI)
                rcl = widgets.RadioButtons(options=[m[0] for m in options[Option.RGC]],
                                                description=Option.RGC.value)
                k_best = widgets.IntText(value=1,description='k:',layout=widgets.Layout(width='150px', display='None'))
                alpha = widgets.FloatSlider(value=0.85, description='alpha:',step=0.05,layout=widgets.Layout(display='None'), orientation='horizontal', min=0, max=1)

                param = widgets.HBox([k_best,alpha])
                rcl_box = widgets.VBox()

                def set_param(change):
                        if change['new'] == 'k-best':
                                k_best.layout.display = 'flex'
                                alpha.layout.display = 'None'
                                rcl_box.children = (rcl,k_best)
                        if change['new'] == 'alpha':
                                k_best.layout.display = 'None'
                                alpha.layout.display = 'flex'
                                rcl_box.children = (rcl,alpha)

                rcl.observe(set_param, names='value')
                self.optionsHandles[Option.RGC] = rcl_box
                set_param({'new':rcl.value})

                return (li_box,rcl_box)


        # helper functions to create widget box for li/sh neighborhoods
        def get_neighborhood_options(self, options: dict, phase: Option):
                available = widgets.Dropdown(
                                options = [m[0] for m in options[phase]],
                                description = phase.value
                )


                size = widgets.IntText(value=1,description='k: ',layout=widgets.Layout(width='150px'))
                add = widgets.Button(description='',icon='chevron-right',layout=widgets.Layout(width='60px'), tooltip='add ' + phase.name)
                remove = widgets.Button(description='',icon='chevron-left',layout=widgets.Layout(width='60px'), tooltip='remove ' + phase.name)
                up = widgets.Button(description='',icon='chevron-up',layout=widgets.Layout(width='30px'), tooltip='up ' + phase.name)
                down = widgets.Button(description='',icon='chevron-down',layout=widgets.Layout(width='30px'), tooltip='down ' + phase.name)
                
                def on_change_available(event):
                        opt = handler.get_options(Problem(self.problemWidget.value),Algorithm(self.algoWidget.value))
                        opt = [o for o in opt.get(phase) if o[0]==available.value][0]
                        size.value = 1 if opt[1] == None or type(opt[1]) == type else opt[1]
                        size.disabled = type(opt[1]) != type or opt[1] == None
                on_change_available(None)

                available.observe(on_change_available, names='value')

                selected = widgets.Select(
                                options = [],
                                description = 'Selected'
                )

                self.optionsHandles[phase] = selected

                add.on_click(self.on_add_neighborhood)
                remove.on_click(self.on_remove_neighborhood)
                up.on_click(self.on_up_neighborhood)
                down.on_click(self.on_down_neighborhood)

                middle = widgets.Box([size, add, remove],layout=widgets.Layout(display='flex',flex_flow='column',align_items='flex-end'))
                sort = widgets.VBox([up,down])
                
                return widgets.HBox([available,middle,selected,sort])        

                
        def on_add_neighborhood(self,event):
                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]
                selected = n_block.children[2]
                size = n_block.children[1].children[0].value
                sel = n_block.children[0].value
                selected.options += (f'{sel}, k={max(1,size)}',)
                selected.index = len(selected.options) -1


        def on_remove_neighborhood(self,event):
                selected = self.get_selected_nh(event)
                
                if len(selected.options) == 0:
                        return

                to_remove = selected.index
                options = list(selected.options)
                del options[to_remove]
                selected.options = tuple(options)

        def on_up_neighborhood(self,event):
                selected = self.get_selected_nh(event)

                if len(selected.options) == 0:
                        return

                to_up = selected.index
                if to_up == 0:
                        return
                options = list(selected.options)
                options[to_up -1], options[to_up] = options[to_up], options[to_up-1]
                selected.options = tuple(options)
                selected.index = to_up -1

        def on_down_neighborhood(self,event):
                selected = self.get_selected_nh(event)

                if len(selected.options) == 0:
                        return

                to_down = selected.index
                if to_down == (len(selected.options) - 1):
                        return
                options = list(selected.options)
                options[to_down +1], options[to_down] = options[to_down], options[to_down+1]
                selected.options = tuple(options)
                selected.index = to_down +1

        def get_selected_nh(self, event):
                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]
                return n_block.children[2]
                
                        



class InterfaceRuntimeAnalysis(InterfaceVisualisation):

        def __init__(self):
                super().__init__(visualisation=False)
                #self.add_instance = widgets.Button(description='Add configuration')
                self.configurations = {}
                #self.selectedConfigs = widgets.Select(options=[])
                self.out = widgets.Output()
                #self.data_available = {}
                self.iteration_df = pd.DataFrame()
                self.summaries = {}
                self.line_checkboxes = widgets.VBox()
                self.run_button.description = 'Run configuration'
                self.iter_slider = widgets.IntSlider(description='iteration', value=1)
                self.iter_slider.layout.display = 'None'
                plt.rcParams['axes.spines.left'] = True
                plt.rcParams['axes.spines.right'] = True
                plt.rcParams['axes.spines.top'] = True
                plt.rcParams['axes.spines.bottom'] = True


        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                #self.add_instance.on_click(self.on_add_instance)
                self.run_button.on_click(self.run)
                self.settingsWidget.children[0].children += (widgets.IntText(value=5, description='runs'), widgets.Checkbox(value=False, description='use previously generated runs'))
                self.settingsWidget.children[0].children[0].value = 100
                reset = widgets.Button(description='Reset', icon='close',layout=widgets.Layout(width='100px',justify_self='end'))
                save_selected = widgets.Button(description='Save selected runs',layout=widgets.Layout(width='150px',justify_self='end'))

                def on_reset(event):
                        self.settingsWidget.children[0].children[0].value = 100
                        self.settingsWidget.children[0].children[1].value = 0
                        self.settingsWidget.children[0].children[2].value = 5
                        self.configurations = {}
                        #self.selectedConfigs.options = []
                        self.problemWidget.disabled = False
                        self.instanceWidget.disabled = False
                        self.settingsWidget.children[0].children[0].disabled = False
                        self.settingsWidget.children[0].children[2].disabled = False
                        self.on_change_algo(None)
                        self.out.clear_output()
                        plt.close()
                        #self.data_available = {}
                        self.iteration_df = pd.DataFrame()
                        self.summaries = {}
                        self.line_checkboxes.children = []
                        self.iter_slider.layout.display= 'None'

                def on_change_iter(change):
                        i = change.new if change.new > 0 else 1
                        self.plot_comparison(i)
                self.iter_slider.observe(on_change_iter,names='value')
        

                reset.on_click(on_reset)
                save_selected.on_click(self.save_runs)

                display(widgets.VBox([self.settingsWidget,self.problemWidget,self.instanceWidget,self.algoWidget,self.optionsWidget,
                        #self.add_instance,self.selectedConfigs, 
                        self.run_button, self.line_checkboxes, widgets.HBox([save_selected,reset]),self.iter_slider]))

                display(self.out)

        def save_runs(self,event):

                checked = {config.description for config in self.line_checkboxes.children if config.value}
                not_saved = {name for name,config in self.configurations.items() if config['settings']['runs'] > len(config['saved'])}
                to_save = checked.intersection(not_saved)

                for s in to_save:
                        config = self.configurations[s]
                        description = self.create_configuration_description(config)
                        name = s[s.find('.')+1:].strip()
                        filepath = self.configuration_exists_in_saved(name, description)

                        use_log = config['settings']['use previously generated runs']
                        runs = config['settings']['runs']
                        seed = config['settings']['seed']

                        if filepath:
                                if seed == 0:
                                        self.save_to_logfile(config,filepath,append=True)
                                        return
                                elif use_log and runs <= len(config['saved']):
                                        # do nothing, only existing runs were loaded
                                        return
                                else:
                                        # write everything to existing file
                                        self.save_to_logfile(config,filepath)
                                        return
                        # create new file and write to it
                        filepath = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + config['prob'].name.lower() + os.path.sep +\
                                        name + '_' + time.strftime('_%Y%m%d_%H%M%S') + '.log'
                        self.save_to_logfile(config,filepath,description=description)

                                        
                                
        def save_to_logfile(self, config: dict, filepath: str, description: str=None, append: bool=False):
                mode = 'w' if description else 'r+'
                f = open(filepath, mode)

                if description:
                        f.write(description+'\n')
                        df = self.iteration_df[config['name']].T
                        df.to_csv(f,sep=' ',na_rep='NaN', mode='a', line_terminator='\n')
                        f.write('S summary\n')
                        self.summaries[config['name']].to_csv(f,na_rep='NaN', sep=' ',mode='a',line_terminator='\n')
                else:
                        saved_runs = set(config['saved'])
                        runs = set(range(1,config['settings']['runs']+1))
                        to_save = list(runs - saved_runs)
                        to_save.sort()
                        if len(to_save) == 0:
                                return
                        data = f.readlines()
                        df = self.iteration_df[config['name']][to_save].T
                        sm = self.summaries[config['name']].loc[to_save]
                        if append: #seed==0
                                existing_runs =int(data[0].split('=')[1].strip())
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
                                
                        else: # seed!= 0
                                data[0] = f"R runs={config['settings']['runs']}\n"
                                idx = next((i for i,v in enumerate(data) if not v[0] in ['R','D']), 0)
                                data = data[:idx]
                                data += [df.to_csv(sep=' ',line_terminator='\n')]
                                data += ['S summary\n']
                                data += [sm.to_csv(sep=' ',line_terminator='\n')]
                                f.seek(0)
                                f.writelines(data)
                                f.truncate()
                                
                f.close()
                self.configurations[config['name']].update({'saved':list(range(1,config['settings']['runs']+1))})


        def configuration_exists_in_saved(self, name: str, description: str):
                description = description[description.find('D '):]
                path = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + Problem(self.problemWidget.value).name.lower()
                log_files = os.listdir(path)
                for l in log_files:
                        if l.startswith(name):
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
        

        def create_configuration_description(self, config: dict):
                s = f"R runs={config['settings']['runs']}\n"
                s += f'D inst={config["inst"]}\n'
                s += f"D seed={config['settings']['seed']}\n"
                s += f"D iterations={config['settings']['iterations']}\n"

                for o in config.get(Option.CH,[]):
                        s += f'D CH{o}\n'
                for o in config.get(Option.LI,[]):
                        s += f'D LI{o}\n'
                for o in config.get(Option.SH,[]):
                        s += f'D SH{o}\n'
                for o in config.get(Option.RGC,[]):
                        s += f'D RGC{o}\n'

                return s.strip()
                
                
        def init_checkbox(self,name: str):
                def on_change(change):
                        self.iter_slider.value = self.get_best_idx()
                        self.plot_comparison(self.iter_slider.value)
                        
                cb = widgets.Checkbox(description=name, value=True)
                cb.observe(on_change,names='value')
                return cb
        

        def run(self, event):
                text = widgets.Label(value='running...')
                display(text)
                # disable prob + inst + iteration + run button
                self.problemWidget.disabled = self.instanceWidget.disabled = True
                self.settingsWidget.children[0].children[0].disabled = True
                self.run_button.disabled = True

                # prepare params and name, save params in dict of configurations
                params = self.prepare_parameters()

                # run algorithm with params or load data from file
                log_df,summary = self.load_datafile_or_run_algorithm(params)

                text.layout.display = 'None'
                self.run_button.disabled = False

                self.iteration_df = pd.concat([self.iteration_df,log_df], axis=1)
                self.summaries[params['name']] = summary

                # add name to checkbox list
                self.line_checkboxes.children +=(self.init_checkbox(params['name']),)
                self.iter_slider.layout.display = 'flex'

                # plot checked data
                self.iter_slider.value = self.get_best_idx()
                self.iter_slider.max = len(self.iteration_df)
                self.plot_comparison(self.iter_slider.value)


        def load_datafile_or_run_algorithm(self,params: dict):
                settings = params['settings']
                if settings['use previously generated runs']:
                        name = params['name'][params['name'].find('.')+1:].strip()
                        description = self.create_configuration_description(params)
                        file_name = self.configuration_exists_in_saved(name,description)
                        if file_name:
                                f = open(file_name, 'r')
                                ex_runs = int(f.readline().split('=')[1].strip())
                                if settings['runs'] <= ex_runs:
                                        data, sm = self.load_datafile(file_name,settings['runs'])
                                        data.columns = pd.MultiIndex.from_tuples(zip([params['name']]*len(data.columns), data.columns))
                                        self.configurations[params['name']].update({'saved':list(data.columns.get_level_values(1))})
                                        return data, sm
                                if settings['seed'] == 0:
                                        runs = settings['runs']
                                        # load existing runs
                                        data, sm = self.load_datafile(file_name,ex_runs)
                                        data.columns = pd.MultiIndex.from_tuples(zip([params['name']]*len(data.columns), data.columns))
                                        # generate runs-ex_runs new ones and set correct run numbers
                                        params['settings'].update({'runs':runs-ex_runs})
                                        new_data, new_sm = handler.run_algorithm_comparison(params)
                                        params['settings'].update({'runs':runs})
                                        new_data.columns = pd.MultiIndex.from_tuples([(n,int(r)+ex_runs) for (n,r) in new_data.columns])
                                        new_sm.index = pd.MultiIndex.from_tuples([(int(r)+ex_runs,m) for (r,m) in new_sm.index])

                                        self.configurations[params['name']].update({'saved':list(data.columns.get_level_values(1))})
                                        # concatenate them
                                        return pd.concat([data,new_data],axis=1),pd.concat([sm,new_sm])
                                
                return handler.run_algorithm_comparison(params)

        def load_datafile(self,filename,runs: int):
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


        def get_best_idx(self):
                checked = [c.description for c in self.line_checkboxes.children if c.value]
                if checked == []:
                        return 1
                df = self.iteration_df[checked]
                m = 1
                if Problem(self.problemWidget.value) in [Problem.MAXSAT,Problem.MISP]:
                        m = df.max().max()
                else:
                        m = df.min().min()
                return df.loc[df.isin([m]).any(axis=1)].index.min()


        def prepare_parameters(self):
                params = super().prepare_parameters()
                name = [f'{params.get("algo").name.lower()}']
                for k,v in params.items():
                        if not type(k) == Option or len(v) == 0:
                                continue
                        o = k.name.lower()+ '-' + '-'.join([str(p[1]) for p in v])
                        o = o.replace('.','')
                        name += [o]

                count = len(self.line_checkboxes.children) + 1
                params['name'] = str(count) + '. ' + '_'.join(name)
                params['saved'] = []
                self.configurations[params['name']] = params
                return params


        def plot_comparison(self, i):
                with self.out:
                        fig = plt.figure(num=f'{self.problemWidget.value}',clear=True)
                        g = gs.GridSpec(2,2)
                        ax = fig.add_subplot(g[0,:])
                        ax_bb = fig.add_subplot(g[1,0])
                        ax_sum = fig.add_subplot(g[1,1])
                        
                        legend_handles=[]
                        checked = [c.description for c in self.line_checkboxes.children if c.value]
                        if checked == []:
                                return
                        for i,c in enumerate(checked):
                                
                                col = f'C{int(c.split(".")[0]) % 10}'
                                maxi = self.iteration_df[c].max(axis=1)
                                mini = self.iteration_df[c].min(axis=1)
                                maxi.plot(color=col, ax=ax)
                                mini.plot(color=col, ax=ax)
                                ax.fill_between(self.iteration_df.index, maxi, mini, where=maxi > mini , facecolor=col, alpha=0.2, interpolate=True)
                                legend_handles += [Line2D([0],[0],color=col,label=c + f' (n={len(self.iteration_df[c].columns)})')]
                        loc = ''
                        best = None
                        selected_data = self.iteration_df[checked]
                        if Problem(self.problemWidget.value) in [Problem.MAXSAT,Problem.MISP]:
                                best = selected_data.cummax(axis=0).cummax(axis=1).iloc[:,-1:]
                                loc = 'lower right'
                        else:
                                best = selected_data.cummin(axis=0).cummin(axis=1).iloc[:,-1:]
                                loc = 'upper right'
                        best.plot(color='black',ax=ax)
                        legend_handles += [Line2D([0],[0],color='black',label='best')]
                        ax.legend(handles=legend_handles,loc=loc)

                        ax.axvline(x=self.iter_slider.value)

                        #create boxplot
                        col = selected_data.columns
                        col = [c for c in col if not c[1] in ['max','min']]
                        bb_data = selected_data[[c for c in col if not c[1] in ['max','min']]]
                        bb_data = bb_data.loc[self.iter_slider.value].reset_index(level=1, drop=True).reset_index()
                        bb_data = bb_data.rename(columns={self.iter_slider.value:f'iteration={self.iter_slider.value}'})
                        bb_data.boxplot(by='index',rot=25,ax=ax_bb)
                        fig.suptitle('')
                        ax_bb.set_xlabel('')
                        ax_bb.set_ylabel('objective value')
                        widgets.interaction.show_inline_matplotlib_plots()

        
        



        
        




    