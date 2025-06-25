from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [Extension('PrepareBatchGraph', sources = ['./FINDER_percolation/PrepareBatchGraph.pyx','./FINDER_percolation/src/lib/PrepareBatchGraph.cpp','./FINDER_percolation/src/lib/graph.cpp','./FINDER_percolation/src/lib/graph_struct.cpp'],language='c++', extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER_percolation/graph.pyx', './FINDER_percolation/src/lib/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER_percolation/utils.pyx', './FINDER_percolation/src/lib/utils.cpp', './FINDER_percolation/src/lib/graph.cpp', './FINDER_percolation/src/lib/graph_utils.cpp', './FINDER_percolation/src/lib/disjoint_set.cpp', './FINDER_percolation/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER_percolation/mvc_env.pyx', './FINDER_percolation/src/lib/mvc_env.cpp', './FINDER_percolation/src/lib/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER_percolation/nstep_replay_mem.pyx', './FINDER_percolation/src/lib/nstep_replay_mem.cpp', './FINDER_percolation/src/lib/graph.cpp', './FINDER_percolation/src/lib/mvc_env.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER_percolation/graph_struct.pyx', './FINDER_percolation/src/lib/graph_struct.cpp'], language='c++', extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER_percolation/GraphDQN.pyx'])
                   ])