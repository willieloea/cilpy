If I have a library that I've written in python, and I've written docstrings for all my classes and methods. I've used mkdocs as a tool to build a site for documentation, but I'd now like to make API documentation. This is the layout of my repo:
```
vostrowillie ~/uni/cs771/code$ tree
.
├── cilpy
│   ├── compare
│   │   └── __init__.py
│   ├── problem
│   │   ├── cmpb.py
│   │   ├── functions.py
│   │   ├── __init__.py
│   │   └── mpb.py
│   ├── runner.py
│   └── solver
│       ├── chm
│       │   ├── debs_rules.py
│       │   ├── epsilon_constrained.py
│       │   ├── __init__.py
│       │   ├── no_handler.py
│       │   ├── penalty.py
│       │   └── repair.py
│       ├── __init__.py
│       └── solvers
│           ├── ccpso.py
│           ├── ccriga.py
│           ├── pso.py
│           ├── __pycache__
│           │   ├── pso.cpython-313.pyc
│           │   └── toplogy.cpython-313.pyc
│           ├── qpso.py
│           └── toplogy.py
├── cilpy.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── docs
│   └── index.md
├── examples
│   ├── ackley_qpso.out.csv
│   ├── ackley_qpso.py
│   ├── cmpb_ccpso.out.csv
│   ├── cmpb_ccpso.py
│   ├── cmpb_ccriga.out.csv
│   ├── cmpb_ccriga.py
│   ├── cmpb_pso.out.csv
│   ├── cmpb_pso.py
│   ├── cmpb_qpso.out.csv
│   ├── cmpb_qpso.py
│   ├── examples
│   │   ├── sphere_lbest_pso.out.csv
│   │   └── sphere_pso.out.csv
│   ├── sphere_pso.out.csv
│   ├── sphere_pso.py
│   ├── sphere_qpso.out.csv
│   ├── sphere_qpso.py
│   └── visualize_results.tmp.py
├── mkdocs.yml
├── pyproject.toml
├── README.md
├── scripts
│   └── gen_api_pages.py
├── test
│   ├── README.md
│   └── test_types.py
├── tmp.md
└── visualize_mpb.py

14 directories, 50 files
```

How can I make API documentation?