## Changes to existing pymhlib files

### log.py

- an additional parser argument **mh_log_step** was added to set a logfile path which is used for logging step-wise information necessary for visualization
- **init_logger()**: added initialization of the step logger, if no filepath was given, step-wise logging is omitted

### scheduler.py

- **Scheduler.__init__()**: added attribute for step logger
- **Scheduler.perform_method()**: added logging of detailed information about current solution before and after the scheduler method is performed, but only if the step logger was initialized (i.e. the logger has handlers)


### solution.py

- added parser arguments **mh_grc_k, mh_grc_alpha, mh_grc_par** to pass information about parameters used in greedy randomized construction
- **Solution.__init__()**: added attribute for step logger
- **Solution** class: 
    - added methods for performing GRASP
        - **greedy_randomized_construction()**: performs all steps of a greedy randomized construction, also includes step-wise logging if the step logger is active
        - **restricted_candidate_list()**: returns an array of solution elements selected from the candidate list, calls selection functions according to the setting parameters for greedy randomized construction
        - methods which are called by greedy_randomized_construction() and restricted_candidate_list(), they have to be overwritten by problem specific implementations
            - **copy_empty()**: returns a copy of the given solution but with an empty solution attribute
            - **is_complete_solution()**: returns *True* if the given solution is complete, *False* otherwise
            - **candidate_list()**: returns a dictionary, keys = solution elements, values = problem specific value of solution element
            - **update_solution()**: adds the element which was randomly selected from the restricted candidate list to the solution and updates the candidate list
            - **restricted_candidate_list_k()**: selects k best elements from a candidate list
            - **restricted_candidate_list_alpha()**: selects elements according to some threshold calculated by using parameter alpha
    - **construct_greedy()**: implements a greedy construction heuristic which can be used to obtain an initial solution (uses same methods as greedy randomized construction)
    - **is_tabu()**: returns True if a solution is forbidden according to givent tabu list, solution specific implementation is necessary
    - **get_tabu_attribute()**: given an old and a new solution this method returns the attribute which is tabu for some iterations, solution specific implementation is necessary


### binvec_solution.py
- **BinaryVectorSolution**: implemtation of methods for Tabu Search
    - **k_flip_neighborhood_search()**: added arguments *tabu_list* and *incument*; adapted the existing method to take into account the tabu list, if given, to perform a resticted neighborhood search (Tabu Search)
    - **is_tabu()**: determines if a solution is tabu; the current implementation evaluates the entire solution, but the proposed move is provided as function parameter in case the implementation is changed to only evaluate the move
    - **get_tabu_attribute()**: determines the tabu attribute given an old and a new solution; the tabu attribute is returned as set of variable assignments, e.g. the tabu attribute {-2,3} forbids an assignment of x2=False and x3=True


### subsetvec_solution.py
- **SubsetVectorSolution**: implemtation of methods for Tabu Search
    - **two_exchange_random_fill_neighborhood_search()**: added arguments *tabu_list* and *incument*; adapted the existing method to take into account the tabu list, if given, to perform a resticted neighborhood search (Tabu Search)
    - **is_tabu()**: determines if a solution is tabu; the current implementation evaluates the entire solution, but the proposed move is provided as function parameter in case the implementation is changed to only evaluate the move
    - **get_tabu_attribute()**: determines the tabu attribute given an old and a new solution; the tabu attribute is returned as set of nodes, e.g. tabu attribute {5} forbids solutions which add node 5 to the independent set


### demos.maxsat.py
- added parser argument *mh_tie_breaking_maxsat* to determine tie breaking method
- **MAXSATSolution**: 
    - implemented problem specific methods for GRASP: restricted_candidate_list_k(), restricted_candidate_list_alpha(), update_solution_and_cl(), copy_empty(), is_complete_solution(), candidate_list()
    - **local_improve_restricted()**: wrapper method for k-flip neighborhood which uses a tabu list
    - **is_better**: overwrites the parent method in order to use different tie breaking methods by calling *is_better_tie*
    - **is_better_tie**: prepared method for implementing different tie breaking methods (currently only standard tie breaking available)


### demos.misp.py
- added parser argument *mh_tie_breaking_misp* to determine tie breaking method
- **MISPSolution**:
    - implemented problem specific methods for GRASP: restricted_candidate_list_k(), restricted_candidate_list_alpha(), update_solution_and_cl(), copy_empty(), is_complete_solution(), candidate_list()
    - **local_improve_restricted()**: wrapper method for two-exchange-random-fill neighborhood search which uses a tabu list
    - **is_better**: overwrites the parent method in order to use different tie breaking methods by calling *is_better_tie*
    - **is_better_tie**: method for implementing different tie breaking methods (currently standard tie breaking and greedy tie breaking is available)


## New pymhlib files

### ts.py
implementation of Tabu Search

### ts_helper.py
implementation of helper classes 
- **TabuList**: manages a tabu list (currently implemented as simple list) which holds TabuAttributes
- **TabuAttribute**: holds a tabu attribute and its tabu tenure

### tests.test_ts.py
unit tests for testing tabu search and tabu list methods