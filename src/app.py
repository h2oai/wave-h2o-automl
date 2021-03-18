import os
from h2o_wave import site, ui, app, Q, main, on, handle_on
from .config import *
from .utils import *
import concurrent.futures
import asyncio
import boto3
import smart_open
import re
import base64

LOCAL_TEST = False

# Initialize DAI client
async def init_dai_client(q: Q):
    try:
        q.user.dai_client = driverlessai.Client(
            address=q.user.dai_address,
            token_provider=lambda: q.auth.access_token)
        await show_tables(q)
    except Exception as e:
        if not q.user.dai_client:
            show_error(q, f'No DAI instance found. Please connect via Steam first. {e}', app_config.main_box)
        return


# Initialize app
async def init_app(q: Q):
    q.app.app_icon, = await q.site.upload(['./static/icon.png'])
    global_nav = [
        ui.nav_group('Navigation', items=[
            ui.nav_item(name='#home', label='Home'),
            ui.nav_item(name='#import', label='Import Data'),
            ui.nav_item(name='#dai', label='Build Model'),
            ui.nav_item(name='#mlflow', label='MLFlow'),
        ])]

    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='xs',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('body', direction=ui.ZoneDirection.ROW, size='900px'),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='m',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('body', direction=ui.ZoneDirection.ROW),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='xl',
            width='1700px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('body', direction=ui.ZoneDirection.COLUMN, size='900px'),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title=app_config.title,
                                      subtitle=app_config.subtitle)  # , nav=global_nav)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')


def main_menu(q: Q):
    q.page['main'] = ui.form_card(box='body', items=app_config.items_guide_tab)


# Clean cards before next route
async def clean_cards(q: Q):
    cards_to_clean = ['main', 'import', 'stepper', 'projects']
    for card in cards_to_clean:
        del q.page[card]

def assign_dai_vars(q: Q):
    # Store user values
    if q.args.dai_address:
        q.user.dai_address = q.args.dai_address
    if q.args.dai_username:
        q.user.dai_username = q.args.dai_username
    if q.args.dai_password:
        q.user.dai_password = q.args.dai_password
    if q.args.dai_target:
        q.user.dai_target = q.args.dai_target
    if q.args.dai_cols_to_drop:
        q.user.dai_cols_to_drop = q.args.dai_cols_to_drop
    if q.args.dai_acc:
        q.user.dai_acc = q.args.dai_acc
    if q.args.dai_time:
        q.user.dai_time = q.args.dai_time
    if q.args.dai_int:
        q.user.dai_int = q.args.dai_int
    if q.args.dai_scorer:
        q.user.dai_scorer = q.args.dai_scorer


def assign_db_vars(q: Q):
    if q.args.db_uname:
        q.user.db_uname = q.args.db_uname
    if q.args.db_instance:
        q.user.db_instance = q.args.db_instance
    if q.args.db_key:
        q.user.db_key = q.args.db_key
    if q.args.db_notebook:
        q.user.db_notebook = q.args.db_notebook


@on('next_start')
@on('back_dai')
async def next_start(q: Q):
      assign_dai_vars(q)
      q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database'),
                ui.step(label='Step 2: DAI Settings', icon='Settings'),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook'),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget')]),
            ui.text_xl('Import Data from S3'),
            ui.textbox(name='s3_train_file', label='Train File',
                       value='s3://h2o-public-test-data/smalldata/kaggle/CreditCard/creditcard_train_cat.csv'),
            ui.textbox(name='s3_test_file', label='Test File',
                       value='s3://h2o-public-test-data/smalldata/kaggle/CreditCard/creditcard_test_cat.csv'),
            ui.button(name='next_dai', label='Next', primary=True)
        ]
    )


@on('next_dai')
async def next_dai(q: Q, warning: str = ''):
    # Assign Databricks settings if back was pressed
    assign_db_vars(q)
    # Store user values from previous inputs
    if q.args.s3_train_file:
        q.user.train_filename = q.args.s3_train_file
    if q.args.s3_train_file:
        q.user.test_filename = q.args.s3_test_file

    # App defaults
    if not q.user.dai_acc:
        q.user.dai_acc = 5
    if not q.user.dai_time:
        q.user.dai_time = 3
    if not q.user.dai_int:
        q.user.dai_int = 10
    if not q.user.dai_scorer:
        q.user.dai_scorer = 'AUC'
    if not q.user.dai_cols_to_drop:
        q.user.dai_cols_to_drop = []

    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database'),
                ui.step(label='Step 2: DAI Settings', icon='Settings'),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook'),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.progress('Importing data')
        ])
    await q.page.save()
    q.user.train_df = pd.read_csv(smart_open.smart_open(q.user.train_filename))
    q.user.test_df = pd.read_csv(smart_open.smart_open(q.user.test_filename))

    target_choices = [ui.choice(i, i) for i in q.user.train_df.columns]
    scorer_list_classification = ['Accuracy', 'AUC', 'AUCPR', 'F05', 'F1', 'F2', 'GINI', 'LOGLOSS', 'MACROAUC']
    scorer_list_regression = ['MAE', 'MAPE', 'MER', 'MSE', 'R2', 'R2COD', 'RMSE', 'RMSLE', 'RMSPE', 'SMAPE']
    scorer_list = scorer_list_classification + scorer_list_regression
    scorer_choices = [ui.choice(i, i) for i in scorer_list]
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings'),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook'),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.separator('Instance Settings'),
            ui.message_bar('danger', warning),
            ui.textbox(name='dai_address', label='DAI Address', required=True, value=q.user.dai_address),
            ui.textbox(name='dai_username', label='DAI Username', required=True, value=q.user.dai_username),
            ui.textbox(name='dai_password', label='DAI Password', required=True, password=True,
                       value=q.user.dai_password),
            ui.separator('Experiment Settings'),
            ui.dropdown(name='dai_target', label='DAI Target', required=True, choices=target_choices,
                        value=q.user.dai_target),
            ui.dropdown(name='dai_cols_to_drop', label='Columns to Drop', choices=target_choices,
                        values=q.user.dai_cols_to_drop),
            ui.slider(name='dai_acc', label='Accuracy', value=q.user.dai_acc, min=1, step=1, max=10),
            ui.slider(name='dai_time', label='Time', value=q.user.dai_time, min=1, step=1, max=10),
            ui.slider(name='dai_int', label='Interpretability', value=q.user.dai_int, min=1, step=1, max=10),
            ui.dropdown(name='dai_scorer', label='Scorer', choices=scorer_choices, value=q.user.dai_scorer),
            ui.buttons([ui.button(name='next_done', label='Next', primary=True),
                        ui.button(name='back_dai', label='Back', primary=False)
                        ])
        ])



# Modify the template file with the user defined settings
def modify_template(replace_dict):
    fp = open("./src/db_template_new.ipynb", "r")
    fp_out = open("./wave_databricks_notebook.ipynb", "w")
    for line in fp:
        line_found = False
        # Replace line if key found
        for k, v in replace_dict.items():
            if k in line:
                # For ints and lists
                if re.match(r'.*(WAVE_DAI_ACC|WAVE_DAI_TIME|WAVE_DAI_INT|WAVE_DAI_COLS2DROP)', line):
                    fp_out.write(line.replace(k, str(v)))
                # Strings
                else:
                    fp_out.write(line.replace(k, "'" + v + "'"))
                line_found = True
        if not line_found:
            fp_out.write(line)
    fp.close()
    fp_out.close()


@on()
async def next_done(q: Q):
    assign_dai_vars(q)

    if not q.args.dai_address or not q.args.dai_username or not q.args.dai_password or not q.args.dai_target:
        await next_dai(q, 'Enter required information')
        return

    replace_dict = {
        'WAVE_DAI_ADDRESS': q.user.dai_address,
        'WAVE_DAI_USERNAME': q.user.dai_username,
        'WAVE_DAI_PASSWORD': q.user.dai_password,
        'WAVE_TRAIN_FILE_PATH': q.user.train_filename,
        'WAVE_TEST_FILE_PATH': q.user.test_filename,
        'WAVE_DAI_TARGET': q.user.dai_target,
        'WAVE_DAI_COLS2DROP': q.user.dai_cols_to_drop,
        'WAVE_DAI_ACC': q.user.dai_acc,
        'WAVE_DAI_TIME': q.user.dai_time,
        'WAVE_DAI_INTERPRETABILITY': q.user.dai_int,
        'WAVE_DAI_SCORER': q.user.dai_scorer,
    }

    # Modify template with user settings
    modify_template(replace_dict)
    # Replace spaces with _
    if q.user.db_notebook:
        q.user.db_notebook = q.user.db_notebook.replace(' ', '_')

    cwd = os.getcwd()
    filepath = cwd + '/wave_databricks_notebook.zip'

    # Zip file for download
    if LOCAL_TEST:
        os.system(f'./venv/bin/zip-files -o {filepath} ./wave_databricks_notebook.ipynb')
    else:
        os.system(f'zip-files -o {filepath} ./wave_databricks_notebook.ipynb')

    download_path, = await q.site.upload([filepath])

    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.message_bar('success', 'Notebook generated successfully!'),
            ui.text(f'[Download processed notebook]({download_path})'),
            ui.buttons([
                ui.button(name='next_view_model_menu', label='View Model', primary=True),
                ui.button(name='next_db', label='Send notebook to Databricks cluster', primary=False),
                ui.button(name='next_start', label='Generate another notebook', primary=False),
            ])
            ])


@on('next_db')
@on('back_db')
async def next_db(q: Q, warning: str = ''):

    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook'),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.message_bar('danger', warning),
            ui.textbox(name='db_uname', label='Databricks Username', required=True, value=q.user.db_uname),
            ui.textbox(name='db_instance', label='Databricks Workspace', required=True,
                       value=q.user.db_instance),
            ui.textbox(name='db_key', label='Databricks Personal Access token', required=True, password=True,
                       value=q.user.db_key),
            ui.link(label='CLICK HERE to view Instructions for obtaining a Token',
                    path='https://docs.databricks.com/dev-tools/api/latest/authentication.html', target=''),
            ui.textbox(name='db_notebook', label='Databricks Notebook Name', required=True, value=q.user.db_notebook),
            ui.buttons([ui.button(name='next_send_to_cluster', label='Next', primary=True),
                        ui.button(name='back_db', label='Back', primary=False)
                        ])
        ])


@on()
async def next_send_to_cluster(q: Q):
    assign_db_vars(q)

    # Get full file path for import
    cwd = os.getcwd()
    filepath = cwd + '/wave_databricks_notebook.ipynb'

    curl_cmd = f"""
    curl -H 'Authorization: Bearer {q.user.db_key}' -F path=/Users/{q.user.db_uname}/{q.user.db_notebook} -F content=@{filepath} \
     -F format=JUPYTER  {q.user.db_instance}/api/2.0/workspace/import
    """
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook'),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.progress('Sending notebook to Databricks cluster'),
        ])

    await q.page.save()
    status = os.system(curl_cmd)
    print(status)
    # Success status is 0
    if status == 0:
        q.page['main'] = ui.form_card(
            box='body',
            items=[
                ui.stepper(name='icon-stepper', items=[
                    ui.step(label='Step 1: Import Data', icon='Database', done=True),
                    ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                    ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                    ui.step(label='Step 4: View Model', icon='ModelingView'),
                    ui.step(label='Step 5: Score', icon='BullseyeTarget'),
                ]),
                ui.message_bar('success', 'Notebook pushed to Databricks cluster successfully!'),
                ui.buttons([
                    ui.button(name='next_view_model_menu', label='View Model', primary=True),
                    ui.button(name='next_start', label='Generate another notebook', primary=False)
                ])
            ])

    else:
        q.page['main'] = ui.form_card(
            box='body',
            items=[
                ui.stepper(name='icon-stepper', items=[
                    ui.step(label='Step 1: Import Data', icon='Database', done=True),
                    ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                    ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                    ui.step(label='Step 4: View Model', icon='ModelingView'),
                    ui.step(label='Step 5: Score', icon='BullseyeTarget'),
                ]),
                ui.message_bar('warning', 'Failed to push notebook. Please check credentials'),
                ui.buttons([
                    ui.button(name='next_start', label='Generate another notebook', primary=True),
                    ui.button(name='back_db', label='Back', primary=False)
                    ])

            ])


@on('next_view_model_menu')
@on('back_view_model_menu')
async def next_view_model_menu(q: Q, warning: str = ''):
    if not q.user.mlflow_url:
        q.user.mlflow_url = 'https://adb-1638020913839231.11.azuredatabricks.net/?o=1638020913839231#mlflow/experiments/1272602360004101'
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.message_bar('danger', warning),
            ui.textbox(name='mlflow_url', label='MLFlow URL', required=True, value=q.user.mlflow_url),
            ui.buttons([ui.button(name='next_view_model', label='View Model', primary=True),
                        ui.button(name='next_start', label='Generate another notebook', primary=False),
                        ])
        ])


@on()
async def next_view_model(q: Q, warning: str = ''):
    del q.page['mlflow']
    if q.args.mlflow_url:
        q.user.mlflow_url = q.args.mlflow_url

    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                ui.step(label='Step 4: View Model', icon='ModelingView'),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.buttons([ui.button(name='next_score_menu', label='Score', primary=True),
                        ui.button(name='back_view_model_menu', label='Back', primary=False),
                        ])
        ])
    q.page['mlflow'] = ui.frame_card(box='body', title='MLFlow', path=q.user.mlflow_url)


@on()
async def next_score_menu(q: Q, warning: str = ''):
    del q.page['mlflow']
    score_choices = [ui.choice(i, i) for i in ['MLFlow', 'External Scorer']]

    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import Data', icon='Database', done=True),
                ui.step(label='Step 2: DAI Settings', icon='Settings', done=True),
                ui.step(label='Step 3: Export Notebook', icon='DietPlanNotebook', done=True),
                ui.step(label='Step 4: View Model', icon='ModelingView', done=True),
                ui.step(label='Step 5: Score', icon='BullseyeTarget'),
            ]),
            ui.message_bar('danger', warning),
            ui.dropdown(name='scorer_choice', label='Scorer Choice', required=True, choices=score_choices, value='MLFlow'),
            ui.buttons([ui.button(name='next_score', label='Score', primary=True),
                        ui.button(name='next_start', label='Generate another notebook', primary=False),
                        ])
        ])



# Main loop
@app('/')
async def serve(q: Q):
    await clean_cards(q)
    if not q.app.initialized:
        await init_app(q)
        q.app.initialized = True

    if not await handle_on(q):
        main_menu(q)
    await q.page.save()
