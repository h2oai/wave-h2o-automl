from h2o_wave import ui
from collections import defaultdict
import os

uploaded_files_dict = defaultdict()
# Card placements across the app
# Order of elements -> offset left, offset top, width, height

logo_file = 'wave_logo.png'
cur_dir = os.getcwd()
uploaded_files_dict['credit_card_train.csv'] = [f'{cur_dir}/data/credit_card_train.csv']
uploaded_files_dict['credit_card_test.csv'] = [f'{cur_dir}/data/credit_card_test.csv']

class Configuration:
    """
    Configuration file Data Labeling
    """
    def __init__(self, env='prod'):
        self._env = env
        self.title = 'H2O AutoML'
        self.subtitle = 'Wave UI for H2O-3 AutoML'
        self.icon = 'Settings'
        self.icon_color = '$yellow'
        self.default_title = ui.text_xl('H2O-3 UI')
        self.items_guide_tab = [
            ui.text("""
<center><img width="240" height=240" src="https://i.imgur.com/jLrt5mr.png"></center>"""),
            ui.frame(content='<h2><center>H2O-3 AutoML</center></h2>', height='60px'),
            ui.text("""
This Wave application demonstrates how to use H2O-3 AutoML via the Wave UI. 
### **Features**:
* **AutoML Training**: Allows a user to train a model using H2O-3 AutoML on custom train/test datasets.<br>
* **Leaderboard**: Visualizing the AML leaderboard.<br>
* **Explainability**: Shows feature importance and row Shapley contributions. <br>
* **Deployment**: Select a model for MOJO download.<br>

Reference: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
            """),
        ]
        self.tmp_dir = '/tmp'
