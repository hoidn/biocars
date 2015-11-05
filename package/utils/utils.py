import os
import dill
import collections
import pdb
import atexit
import numpy as np
import hashlib
from pathos.multiprocessing import ProcessingPool
from pathos import multiprocessing

#import signal
#import sys
#import time
#
#def signal_handler(signal, frame):
#    print 'You pressed Ctrl+C!'
#    sys.exit(0)
#
#signal.signal(signal.SIGINT, signal_handler)

#def accumulator(func, accum, lst):
#    """
#    higher-order function to perform accumulation
#    """
#    if len(lst) == 0:
#        return accum
#    else:
#        return accumulator(func, func(accum, lst[0]), lst[1:])

def parallelmap(func, data, nodes = None):
    """
    Return the averaged signal and background (based on blank frames) over the given runs
    """
    if not nodes:
        nodes = multiprocessing.cpu_count() - 2
    pool = ProcessingPool(nodes=nodes)
    try:
        return pool.map(func, data)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

#def extrap1d(interpolator):
#    xs = interpolator.x
#    ys = interpolator.y
#
#    def pointwise(x):
#        if x < xs[0]:
#            return 0.
#        elif x > xs[-1]:
#            return 0.
#        else:
#            return interpolator(x)
#
#    def ufunclike(xs):
#        try:
#            iter(xs)
#        except TypeError:
#            xs = np.array([xs])
#        return np.array(map(pointwise, np.array(xs)))
#    return ufunclike

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return 0.
        elif x > xs[-1]:
            return 0.
        else:
            return interpolator(x)

    def ufunclike(x):
        ys = np.zeros((len(x)))
        good_indices = np.where(np.logical_and(x >= xs[0], x <= xs[-1]))[0]
        ys[good_indices] = interpolator(x[good_indices])
        return ys
        #return np.array(map(pointwise, np.array(xs)))

    return ufunclike

def make_hashable(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(obj).hexdigest()

def hashable_dict(d):
    """
    try to make a dict convertible into a frozen set by 
    replacing any values that aren't hashable but support the 
    python buffer protocol by their sha1 hashes
    """
    #TODO: replace type check by check for object's bufferability
    for k, v in d.iteritems():
        # for some reason ndarray.__hash__ is defined but is None! very strange
        #if (not isinstance(v, collections.Hashable)) or (not v.__hash__):
        if isinstance(v, np.ndarray):
            d[k] = make_hashable(v)
    return d


def persist_to_file(file_name):
    """
    Decorator for memoizing function calls to disk
    Inputs:
        file_name: File name prefix for the cache file(s)
    """
    # These are the hoops we need to jump through because python doesn't allow
    # assigning to variables in enclosing scope:
    state = {'loaded': False, 'cache_changed': False}
    def check_cache_loaded():
        return state['loaded']
    def flag_cache_loaded():
        state['loaded'] = True
    def check_cache_changed():
        return state['cache_changed']
    def flag_cache_changed():
        return state['cache_changed']

    # Optimization: initialize the cache dict but don't load data from disk
    # until the memoized function is called.
    cache = {}

    def dump():
        os.system('mkdir -p ' + os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            dill.dump(cache, f)

    def decorator(func):
        #check if function is a closure and if so construct a dict of its
        #bindings
        if func.func_code.co_freevars:
            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
        else:
            closure_dict = {}

        def gen_key(*args, **kwargs):
            """
            Based on args and kwargs of a function, as well as the 
            closure bindings, generate a cache lookup key
            """
            return hashlib.sha1(dill.dumps(args)).hexdigest(), hashlib.sha1(dill.dumps(kwargs.items())).hexdigest(), hashlib.sha1(dill.dumps(closure_dict.items())).hexdigest() 

        def compute(*args, **kwargs):
            key = gen_key(*args, **kwargs)
            if not check_cache_loaded():
                try:
                    with open(file_name, 'r') as f:
                        to_load = dill.load(f)
                        print "loading cache"
                        for k, v in to_load.items():
                            cache[k] = v
                except (IOError, ValueError):
                    print "no cache file found"
                flag_cache_loaded()
            if not key in cache.keys():
                cache[key] = func(*args, **kwargs)
                if not check_cache_changed():
                    # write cache to file at interpreter exit if it has been
                    # altered
                    atexit.register(dump)
                    flag_cache_changed()


        def new_func(*args, **kwargs):
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if "flush" in kwargs.keys():
                kwargs.pop("flush", None)
                key = gen_key(*args, **kwargs)
                compute(key)
            key = gen_key(*args, **kwargs)
            if key not in cache:
                compute(*args, **kwargs)
            return cache[key]
        return new_func

    return decorator

# TODO: carry over the recent changes from persist_to_file
def eager_persist_to_file(file_name):
    """
    Decorator for memoizing function calls to disk.
    Differs from persist_to_file in that the cache file is accessed and updated
    at every call, and that each call is cached in a separate file. This allows
    parallelization without problems of concurrency of the memoization cache,
    provided that the decorated function is expensive enough that the
    additional read/write operations have a negligible impact on performance.
    Inputs:
        file_name: File name prefix for the cache file(s)
    """
    cache = {}

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        if func.func_code.co_freevars:
            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
        else:
            closure_dict = {}

        def gen_key(*args, **kwargs):
            """
            Based on args and kwargs of a function, as well as the 
            closure bindings, generate a cache lookup key
            """
            return hashlib.sha1(dill.dumps(args)).hexdigest(), hashlib.sha1(dill.dumps(kwargs.items())).hexdigest(), hashlib.sha1(dill.dumps(closure_dict.items())).hexdigest() 

        def compute(*args, **kwargs):
            local_cache = {}
            file_name = kwargs.pop('file_name', None)
            key = gen_key(*args, **kwargs)
            local_cache[key] = func(*args, **kwargs)
            cache[key] = local_cache[key]
            os.system('mkdir -p ' + os.path.dirname(file_name))
            with open(file_name, 'w') as f:
                dill.dump(local_cache, f)

        def new_func(*args, **kwargs):
            # Because we're splitting into multiple files, we can't retrieve the
            # cache until here
            full_name = file_name + '_' + str(hash(dill.dumps(args)))
            try:
                with open(full_name, 'r') as f:
                    new_cache = dill.load(f)
                    for k, v in new_cache.items():
                        cache[k] = v
            except (IOError, ValueError):
                print "no cache found"
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if "flush" in kwargs.keys():
                kwargs.pop("flush", None)
                # TODO: refactor
                compute(*args, file_name = full_name, **kwargs)
            key = gen_key(*args, **kwargs)
            if key not in cache:
                compute(*args, file_name = full_name, **kwargs)
            return cache[key]
        return new_func

    return decorator
