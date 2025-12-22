'''
code.tem 的 Docstring
筛选所有不包含faithfulness的词
'''

import pandas as pd
import os

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/citations_select"