from h2o_wave import Q, ui, app, main, data
import pandas as pd
from .config import *
from subprocess import Popen, PIPE

# Table from Pandas dataframe
def table_from_df(df: pd.DataFrame, table_name: str, sortable=False, filterable=False, searchable=False, groupable=False):
    # Columns for the table
    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=sortable,  # Make column sortable
        filterable=filterable,  # Make column filterable
        searchable=searchable  # Make column searchable
    ) for x in df.columns.values]
    # Rows for the table
    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()]
    table = ui.table(name=f'{table_name}',
             rows=rows,
             columns=columns,
             multiple=False,  # Allow multiple row selection
             groupable=groupable)
    return table


# MOJO for regression problems
def get_mojo_preds(fname):
    cmd = "java -Dai.h2o.mojos.runtime.license.file="+app_config.scoring_path+"/license.sig -cp "+app_config.scoring_path+"/mojo2-runtime.jar ai.h2o.mojos.ExecuteMojo "+app_config.scoring_path+"/pipeline.mojo "+fname
    process = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stderr)
    rel = str(stdout)[2:-2]
    vals = [float(l.replace("\\","")) for l in rel.split("\\n")[1:]]
    return vals


