import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tbparse import SummaryReader

class ModelFileHandler:
    def __init__(self, file_path):
        assert isinstance(file_path, str), "file must be a string path an event file"
        self.file_path = file_path
        self.reader = SummaryReader(file_path)
        self.df = self.reader.scalars
        self.tags = self.df["tag"].unique()
    
    def get_logged_tags(self, writeout=False):
        tags = self.df["tag"].unique()
        if writeout:
            for t in tags: print(t)
        return tags

    def get_tag_scalars(self, tag: str):
        assert tag in self.tags, f"Tag '{tag}' is not found in {self.file_path}"
        tag_df = self.df[self.df["tag"] == tag]
        return tag_df["step"].to_numpy(), tag_df["value"].to_numpy()

class PlotHandler:
    def __init__(self):
        ...

    def plot(self):
        ...
        

def plot(x, y, ax=None):
    plt.plot(x,y)
    plt.show()