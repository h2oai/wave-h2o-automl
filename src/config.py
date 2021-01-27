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
        self.title = 'H2O MLOps Demo'
        self.subtitle = 'Demo of Steam DAI and MLOps in Wave'
        self.icon = 'OfflineStorageSolid'
        self.icon_color = '$yellow'
        self.default_title = ui.text_xl('MLOps & DAI')
        self.items_guide_tab = [
            ui.text("""
<center><img width="480" height=240" src="https://i.imgur.com/BWcRLg2.jpg"></center>"""),
            ui.frame(content='<h2><center>Geospatial Analytics</center></h2>', height='60px'),
            ui.text("""
This Wave application demonstrates how to analyze Geospatial data.  
### **Features**:
* **Explore**: Interactive table to explore data.<br>
* **Analysis**: Shows dashboards related to tracking data.<br>
* **Insights**: Simple route prediction, clustering and outlier detection using HDBSCAN.<br>

Sample Data: Vessel traffic data can be downloaded from https://marinecadastre.gov/ais/. A sample has been included in this app as well. 
            """),
        ]
        self.banner_box = '1 1 -1 1'
        self.navbar_box = '3 1 3 1'
        self.logo_box = '12 1 -1 1'
        self.main_box = '1 2 -1 -1'
        self.plot11_box = '1 2 5 -1'
        self.plot12_box = '6 2 -1 -1'
        self.plot1_box = '1 2 -1 3'
        self.plot2_box = '1 4 -1 -1'
        self.tmp_dir = '/tmp'

app_config = Configuration()
