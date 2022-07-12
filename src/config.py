from h2o_wave import ui
from collections import defaultdict
import os
import matplotlib

matplotlib.rcParams['figure.dpi'] = 200 # make it nicer on high dpi screens
FIGSIZE=(11, 5)

uploaded_files_dict = defaultdict()
# Card placements across the app
# Order of elements -> offset left, offset top, width, height

logo_file = 'wave_logo.png'
cur_dir = os.getcwd()
uploaded_files_dict['credit_card_train.csv'] = [f'{cur_dir}/data/credit_card_train.csv']
uploaded_files_dict['credit_card_test.csv'] = [f'{cur_dir}/data/credit_card_test.csv']

uploaded_files_dict['wine_quality_train.csv'] = [f'{cur_dir}/data/wine_quality_train.csv']
uploaded_files_dict['wine_quality_test.csv'] = [f'{cur_dir}/data/wine_quality_test.csv']

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
        self.tmp_dir = '/tmp'
