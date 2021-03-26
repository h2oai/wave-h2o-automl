from h2o_wave import ui
from collections import defaultdict
import os

uploaded_files_dict = defaultdict()
logo_file = 'wave_logo.png'
cur_dir = os.getcwd()

class Configuration:
    """
    Configuration file
    """
    def __init__(self):
        self.title = 'H2O Steam Demo'
        self.subtitle = 'Demo of Steam in Wave'
        self.items_guide_tab = [
            #ui.text("""<center><img width="700" height=240" src="https://i.imgur.com/Jv54JjL.png"></center>"""),
            ui.frame(content='<h2><center>H2O Steam</center></h2>', height='60px'),
            ui.text("""
**Detailed Description:** This application utilizes the Steam connected to Wave cloud to demo MLOPs deployment and management.

### **Features**:
* **DAI**: Connect to a users Steam profile and manage DAI instances.<br>
* **H2O-3**: Connect to a users Steam profile and manage H2O-3 instances.<br>

            """),
            ui.buttons([ui.button(name='#steam', label='Get Started', primary=True)], justify='center'),
        ]
        self.tmp_dir = '/tmp'
        self.steam_menu_box = 'body_main'
        self.steam_table_box = 'body_main'


app_config = Configuration()
