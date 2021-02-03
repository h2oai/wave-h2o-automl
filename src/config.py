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
        self.title = 'H2O Steam & MLOps Demo'
        self.subtitle = 'Demo of Steam DAI and MLOps in Wave'
        self.default_title = ui.text_xl('MLOps & DAI')
        self.items_guide_tab = [
            ui.text("""
<center><img width="700" height=240" src="https://i.imgur.com/Jv54JjL.png"></center>"""),
            ui.frame(content='<h2><center>H2O Steam & MLOPs</center></h2>', height='60px'),
            ui.text("""
**Detailed Description:** This application utilizes the Steam and MLOps connected to Wave cloud to demo MLOPs deployment and management.

### **Features**:
* **Steam**: Connect to a users Steam profile and manage DAI instances.<br>
* **DAI**: Allows a user to connect to DAI or export experiments to MLOps.<br>
* **MLOps**: Allows a user to view and delete MLOps projects, deployments and test deployment endpoint for scoring.<br>
            """),
            ui.buttons([ui.button(name='#steam', label='Get Started', primary=True)], justify='center')
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
