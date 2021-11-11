import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class ShowChart:
    def __init__(self):
        return
    
    def show(self, count_path):       #count : xlsx파일
        data = pd.read_excel(count_path)
        data.plot()
    

