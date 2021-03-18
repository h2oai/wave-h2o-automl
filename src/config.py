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
        self.title = 'MOJO to Wave Scorer'
        self.subtitle = ''
        self.items_guide_tab = [
            ui.text("""<center><img width="450" height=300" src="https://i.imgur.com/tUlLW7M.png"></center>"""),
            ui.frame(content='<h2><center>MOJO to Wave Scorer</center></h2>', height='60px'),
            ui.text("""
**Detailed Description:** Use a MOJO generated from Driverless AI to generate a custom scoring application.

### **Features**:
* Import a DAI MOJO and test csv file.<br>
* Creates a custom application with user defined settings and dashboards that can be used for scoring.<br>
            """),
            ui.buttons([ui.button(name='next_start', label='Get Started', primary=True)], justify='center'),
        ]
        self.banner_box = '1 1 -1 1'
        self.navbar_box = '4 1 3 1'
        self.logo_box = '12 1 -1 1'
        self.main_box = '1 2 -1 -1'
        self.plot11_box = '1 2 5 -1'
        self.plot12_box = '6 2 -1 -1'
        self.plot1_box = '1 2 -1 3'
        self.plot2_box = '1 4 -1 -1'
        self.tmp_dir = '/tmp'


app_config = Configuration()
