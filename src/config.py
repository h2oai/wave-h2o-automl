from h2o_wave import ui
from collections import defaultdict
import os

uploaded_files_dict = defaultdict()
cur_dir = os.getcwd()

class Configuration:
    """
    Configuration file Data Labeling
    """
    def __init__(self):
        self.title = 'Hospital Capacity Simulation'
        self.subtitle = 'Hospital bed availability simulation'
        self.icon = 'Nav2DMapView'
        self.icon_color = '$yellow'
        self.default_title = ui.text_xl('Hospital bed availability')
        self.items_guide_tab = [
            ui.frame(content='<h2><center>Hospital Capacity Simulation</center></h2>', height='60px'),
            ui.text("""
            
Hospital capacity simulation and predicting patient stay using a H2O MOJO.

### **Features**:
* **Patient Simulation**: Uses Synthea to generate synthetic patient data with user defined medical conditions.<br>
* **Hospital Stay Prediction**: Uses a H2O DAI MOJO to predict hospital stay.<br>
* **Simulation**: Simulates a hospital environment with user defined settings to predict bed availability.

References:
Synthea: https://synthetichealth.github.io/synthea/ 
            """),
            ui.text("""
<center><img width="800" height=400" src="https://i.imgur.com/kTmRQLz.png"></center>""")
        ]
        self.synthea_img = """
<center><img width="700" height=500" src="https://i.imgur.com/SGBtbWJ.png"></center>"""
        self.banner_box = '1 1 11 1'
        self.logo_box = '12 1 -1 1'
        self.menu_box = '1 2 3 -1'
        self.main_box = '1 2 -1 -1'
        self.small_main_box = '4 2 -1 -1'
        self.button_bar_box = '6 9 1 1'
        self.plot01_box = '1 2 1 1'
        self.plot02_box = '2 2 2 1'
        self.plot03_box = '4 2 1 1'
        self.plot04_box = '5 2 1 1'
        self.plot05_box = '6 2 1 1'
        self.plot06_box = '7 2 1 1'
        self.plot07_box = '8 2 1 1'
        self.plot08_box = '9 2 -1 1'
        self.plot1_box = '1 3 -1 3'
        self.plot21_box = '1 6 6 3'
        self.plot22_box = '7 6 -1 3'
        self.notification_box = '6 3 2 2'
        self.tmp_dir = '/tmp'
        self.scoring_path = './src/scoring'
        self.logo_file = 'wave_logo.png'
        self.app_icon_file = 'hosp.png'
        self.synthea_output = './src/simulators/synthea_output'
        
app_config = Configuration()
