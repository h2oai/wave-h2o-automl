from h2o_wave import ui
from collections import defaultdict
import os

uploaded_files_dict = defaultdict()
logo_file = 'wave_logo.png'
cur_dir = os.getcwd()

class Configuration:
    """
    Configuration file Data Labeling
    """
    def __init__(self):
        self.title = 'Wave Databricks Notebook Generator'
        self.subtitle = 'Generate a notebook using Wave'
        self.default_title = ui.text_xl('Wave DB Notebook')
        self.items_guide_tab = [
            ui.text("""
<center><img width="700" height=240" src="https://i.imgur.com/3FpWucB.png"></center>"""),
            ui.frame(content='<h2><center>Wave Databricks Notebook Generator</center></h2><br><br><br>'),
            ui.text("""
This application allows a user to generate a Databricks notebook based on user defined settings. 
* Specify custom S3 based datasets for training and testing
* Specify Driverless AI (DAI) experiment settings
* Generate a notebook that can be used in the Databricks environment
* Allows a user to import the notebook to a Databricks workspace directly
            """),
            ui.buttons([ui.button(name='next_start', label='Get Started', primary=True)], justify='center'),
        ]
        self.tmp_dir = '/tmp'


app_config = Configuration()
