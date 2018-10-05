# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
If we use percentiles (say, the lower and upper 30%iles) to define La Nina and
El Nino, respectively, MEI ranks from 1-21 denote strong to weak La Nina
conditions, while 48-68 (49-69) denote weak to strong El Nino conditions.
If one uses the quintile definition for (moderate or stronger) ENSO events, MEI
ranks from 1-14 would denote La Nina, while 55-68 (56-69)  would denote El Nino.
Finally, the comparison figures on this website refer to strong ENSO events,
such as might be defined by the top 7 (upper decile) ranks, such as 1-7 for
La Nina, and 62-68 (63-69) for El Nino.
"""

import numpy as np
import pandas as pd


folder = 'c:/Felix/ORA/python_scripts/Test_GITHUB/MEI/'
arc = 'MEI_values.txt'
MEI = pd.read_csv(folder + arc)
rank = MEI.rank()
rank['YEAR'] = MEI['YEAR']
