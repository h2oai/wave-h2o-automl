import h2o
from h2o.automl import H2OAutoML
from h2o_wave import Q, ui, app, main, data
from .config import *
import time
import pandas as pd
import asyncio
import concurrent.futures
import numpy as np
import base64
import io

h2o.init()
app_config = Configuration()

@app('/')
async def serve(q: Q):
    cur_dir = os.getcwd()
    q.app.tmp_dir = cur_dir + app_config.tmp_dir
    if not os.path.exists(q.app.tmp_dir):
        os.mkdir(q.app.tmp_dir)


    # Hash routes user when tabs are clicked
    hash = q.args['#']
    if hash == 'guide':
        await clean_cards(q)
        await main_menu(q)
    elif hash == 'import':
        await clean_cards(q)
        await import_menu(q)
    elif hash == 'train':
        await clean_cards(q)
        await select_table(q)
    elif hash == 'lb' or q.args.back_lb:
        await clean_cards(q)
        await show_lb(q)
    # User selected files from import menu
    elif q.args.uploaded_file:
        await upload_data(q)
    # User selected train/test file
    elif q.args.selected_tables_next:
        await train_menu(q)
    # User starts training
    elif q.args.next_train:
        await train_model(q)
    # Row in table was clicked
    elif q.args.lb_table:
        await get_mojo(q)
    elif q.args.shap_row_index:
        await get_mojo(q)
    else:
        await main_menu(q)
    await q.page.save()


async def main_menu(q: Q):
    q.app.df_rows = []
    q.page['banner'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      icon=app_config.icon, icon_color=app_config.icon_color)

    q.page['menu'] = ui.toolbar_card(
        box=app_config.navbar_box,
        items=[
            ui.command(name="#guide", label="Home", caption="Home", icon="Home"),
            ui.command(name="#import", label="Import Data", caption="Import Data", icon="Database"),
            ui.command(name="#train", label="Train", caption="Train", icon="BullseyeTarget"),
            ui.command(name="#lb", label="Leaderboard", caption="Leaderboard", icon="ClipboardList"),
        ])

    # Logo
    if not q.app.logo_url:
        q.app.logo_url, = await q.site.upload([logo_file])

    # Navbar and main placeholder
    # q.page['logo'] = ui.markup_card(
    #   box=logo_box,
    #  title='',
    #  content="""<p style='text-align:center; vertical-align: middle; display: table-cell; width: 134px;'>"""
    #          """<a href='https://www.h2o.ai/h2o-q/'> <img src='""" + q.app.logo_url + """' height='50px' width='50px'> </a> </p>"""

    # )

    q.page['main'] = ui.form_card(box=app_config.main_box, items=app_config.items_guide_tab)
    await q.page.save()


# Menu for importing new datasets
async def import_menu(q: Q):
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[
        ui.text_xl('Import Data'),
        ui.file_upload(name='uploaded_file', label='Upload File', multiple=True),
    ])


async def upload_data(q: Q):
    uploaded_file_path = q.args.uploaded_file
    for file in uploaded_file_path:
        filename = file.split('/')[-1]
        uploaded_files_dict[filename] = uploaded_file_path
    time.sleep(1)
    q.page['main'] = ui.form_card(box=app_config.main_box,
                                  items=[ui.message_bar('success', 'File Imported! Please select an action'),
                                         ui.buttons([ui.button(name='#train', label='Train a model', primary=True),
                                                     ui.button(name='#guide', label='Main Menu', primary=False)])])


# Menu for selecting a pre-loaded table
async def select_table(q: Q, warning: str = ''):
    choices = []
    if uploaded_files_dict:
        for file in uploaded_files_dict:
            choices.append(ui.choice(file, file))
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.message_bar(type='warning', text=warning),
            ui.dropdown(name='train_file', label='Training data', value=q.app.train_file, required=True,
                        choices=choices),
            ui.dropdown(name='test_file', label='Test data (optional)', value=q.app.test_file, required=False,
                        choices=choices),
            ui.buttons([ui.button(name='selected_tables_next', label='Next', primary=True)])
        ])
    else:
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.text_xl(f'{q.app.task}'),
            ui.message_bar(type='warning', text=warning),
            ui.text(f'No data found. Please import data first.'),
            ui.buttons([ui.button(name='#import', label='Import Data', primary=True)])
        ])


# Settings for training
async def train_menu(q: Q, warning: str = ''):
    # Error handling
    if not q.args.train_file and not q.app.train_file:
        await select_table(q, 'Please select training data')
        return
    # Store train/test file
    if q.args.train_file:
        q.app.train_file = q.args.train_file
        # Check for data provided as part of app
        if 'data/credit_card_train.csv' in uploaded_files_dict[q.app.train_file][0]:
            local_path = uploaded_files_dict[q.app.train_file][0]
        else:
            local_path = await q.site.download(uploaded_files_dict[q.app.train_file][0], '.')
        q.app.train_df = pd.read_csv(local_path)
    if q.args.test_file:
        q.app.test_file = q.args.test_file
        # Check for data provided as part of app
        if 'data/credit_card_test.csv' in uploaded_files_dict[q.app.test_file][0]:
            local_path = uploaded_files_dict[q.app.test_file][0]
        else:
            local_path = await q.site.download(uploaded_files_dict[q.app.test_file][0], '.')
        q.app.test_df = pd.read_csv(local_path)

    # Default options
    if not q.app.max_models:
        q.app.max_models = 10
    if not q.app.is_classification:
        q.app.is_classification = True
    if not q.app.nfolds:
        q.app.nfolds = 5
    if not q.app.es_metric:
        q.app.es_metric = 'AUTO'
    if not q.app.es_rounds:
        q.app.es_rounds = 3

    es_metrics = ['AUTO', 'deviance', 'logloss', 'MSE', 'RMSE', 'MAE', 'RMSLE', 'AUC', 'AUCPR', 'lift_top_group',
                  'misclassification', 'mean_per_class_error']
    es_metrics_choices = [ui.choice(i, i) for i in es_metrics]
    choices = [ui.choice(i, i) for i in list(q.app.train_df.columns)]
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[
        ui.text_xl(f'Training Options'),
        ui.message_bar(type='warning', text=warning),
        ui.dropdown(name='target', label='Target Column', value=q.app.target, required=True, choices=choices),
        ui.toggle(name='is_classification', label='Classification', value=q.app.is_classification),
        ui.separator('Training Parameters'),
        ui.slider(name='max_models', label='Max Models', value=q.app.max_models, min=2, step=1, max=512),
        ui.slider(name='max_runtime_mins', label='Max Runtime (minutes)', value=q.app.max_runtime_secs, min=1, step=1,
                  max=1440),
        ui.dropdown(name='es_metric', label='Early stopping metric', value=q.app.es_metric, required=True,
                    choices=es_metrics_choices),
        ui.textbox(name='es_rounds', label='Early stopping rounds', value=q.app.es_rounds),
        ui.textbox(name='nfolds', label='nfolds', value=q.app.nfolds),
        ui.buttons([ui.button(name='next_train', label='Next', primary=True)])
    ])


# Train progress
async def show_timer(q: Q):
    main_page = q.page['main']
    max_runtime_secs = q.app.max_runtime_mins * 60
    for i in range(1, max_runtime_secs):
        pct_complete = int(np.ceil(i/max_runtime_secs * 100))
        main_page.items = [ui.progress(label='Training Progress', caption=f'{pct_complete}% complete', value=i / max_runtime_secs)]
        await q.page.save()
        await q.sleep(1)


# AML train
def aml_train(aml, x, y, train):
    aml.train(x=x, y=y, training_frame=train)


# Train AML model
async def train_model(q: Q):
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[])
    if not q.args.target:
        await train_menu(q, 'Please select target column')
        return

    q.app.target = q.args.target
    q.app.max_models = q.args.max_models
    q.app.max_runtime_mins = q.args.max_runtime_mins
    q.app.es_metric = q.args.es_metric
    q.app.es_rounds = q.args.es_rounds
    q.app.nfolds = q.args.nfolds
    q.app.is_classification = q.args.is_classification

    y = q.app.target

    main_page = q.page['main']
    main_page.items = [ui.progress(label='Training the model')]
    await q.page.save()

    # Import a sample binary outcome train/test set into H2O
    train = h2o.H2OFrame(q.app.train_df)
    if q.app.test_file:
        test = h2o.H2OFrame(q.app.test_df)
        if q.app.is_classification:
            test[y] = test[y].asfactor()

    # Identify predictors and response
    x = train.columns
    x.remove(y)

    # For binary classification, response should be a factor
    if q.app.is_classification:
        train[y] = train[y].asfactor()

    # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
    max_runtime_secs = q.app.max_runtime_mins * 60
    aml = H2OAutoML(max_models=q.app.max_models, max_runtime_secs=max_runtime_secs, nfolds=q.app.nfolds,
                    stopping_metric=q.app.es_metric, stopping_rounds=q.app.es_rounds, seed=1)

    future = asyncio.ensure_future(show_timer(q))
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await q.exec(pool, aml_train, aml, x, y, train)
    future.cancel()
    q.app.aml = aml
    await show_lb(q)


# Table from Pandas dataframe
def table_from_df(df: pd.DataFrame, table_name: str):
    # Columns for the table
    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=True,  # Make column sortable
        filterable=True,  # Make column filterable
        searchable=True  # Make column searchable
    ) for x in df.columns.values]
    # Rows for the table
    rows = [ui.table_row(name=str(i), cells=[cell for cell in row]) for i, row in df.iterrows()]
    table = ui.table(name=f'{table_name}',
             rows=rows,
             columns=columns,
             multiple=False,  # Allow multiple row selection
             height='100%')
    return table


# Show leaderboard
async def show_lb(q: Q):
    if q.app.aml:
        # H2O automl object
        aml = q.app.aml
        # Get leaderboard
        lb = aml.leaderboard
        lb_df = lb.as_data_frame()

        lb_table = table_from_df(lb_df, 'lb_table')

        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.text_xl('AutoML Leaderboard'),
            ui.text(f'**Training shape:** {q.app.train_df.shape}'),
            ui.text(f'**Target:** {q.app.target}'),
            ui.text_m(f'**Select a model to get the MOJO**'),
            lb_table
        ])
    else:
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.text_xl('AutoML Leaderboard'),
            ui.text('No models trained. Please train a model first.'),
            ui.buttons([ui.button(name='#train', label='Train a model', primary=True)])
        ])


# Clean cards
async def clean_cards(q: Q):
    cards_to_clean = ['plot1', 'plot21', 'plot22']
    for card in cards_to_clean:
        del q.page[card]


# Image from matplotlib object
def get_image_from_matplotlib(matplotlib_obj):
    buffer = io.BytesIO()
    matplotlib_obj.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# Get MOJO for selected row in table
async def get_mojo(q: Q):
    if q.args.lb_table:
        q.app.selected_model = q.args.lb_table[0]
    if not q.app.shap_row_index:
        q.app.shap_row_index = 0
    if q.args.shap_row_index:
        q.app.shap_row_index = q.args.shap_row_index

    main_page = q.page['main']
    main_page.items = [ui.progress(f'Generating model insights')]
    await q.page.save()

    model_index = int(q.app.selected_model)
    # H2O automl object
    aml = q.app.aml
    # Get leaderboard
    lb = aml.leaderboard
    lb_df = lb.as_data_frame()

    model_str = lb_df['model_id'].at[model_index]
    model = h2o.get_model(model_str)
    mojo_path = model.download_mojo(path=f'{q.app.tmp_dir}')

    y = q.app.target

    train = h2o.H2OFrame(q.app.train_df)
    if q.app.test_file:
        test = h2o.H2OFrame(q.app.test_df)
        if q.app.is_classification:
            test[y] = test[y].asfactor()

    download_path, = await q.site.upload([mojo_path])
    shap_choices = [ui.choice(i, i) for i in range(min(10, q.app.train_df.shape[0]))]

    q.page['main'] = ui.form_card(box=app_config.plot1_box, items=[
        ui.text_l(f'**Model:** {model_str}'),
        ui.text(f'[Download MOJO]({download_path})'),
        ui.dropdown(name='shap_row_index', label='Select Row for Shapley Plot', value=q.app.shap_row_index,
                    choices=shap_choices, trigger=True),
        ui.buttons([ui.button(name='back_lb', label='Back to Leaderboard', primary=True)])
    ])

    # Variable importance plots
    try:
        var_imp_df = model.varimp(use_pandas=True)
        sorted_df = var_imp_df.sort_values(by='scaled_importance', ascending=True).iloc[0:10]
        rows = list(zip(sorted_df['variable'], sorted_df['scaled_importance']))
        q.page.add('plot21', ui.plot_card(
            box=app_config.plot21_box,
            title='Variable Importance Plot',
            data=data('feature score', rows=rows),
            plot=ui.plot([ui.mark(type='interval', x='=score', y='=feature', x_min=0, y_min=0, x_title='Relative Importance',
                                  y_title='Feature', color='#33BBFF')])
        ))
    except Exception as e:
        print(f'No var_imp found for {model_str}: {e}')
        q.page['plot21'] = ui.form_card(box=app_config.plot21_box, items=[
            ui.text(f'Variable importance unavailable for **{model_str}**')
           ])
    try:
        shap_plot = model.shap_explain_row_plot(frame=train, row_index=q.app.shap_row_index)
        q.page['plot22'] = ui.image_card(
            box=app_config.plot22_box,
            title="Shapley Plot",
            type="png",
            image=get_image_from_matplotlib(shap_plot),
        )
    except Exception as e:
        print(f'No shap found for {model_str}: {e}')
        q.page['plot22'] = ui.form_card(box=app_config.plot22_box, items=[
            ui.text(f'Shapley unavailable for **{model_str}**')
           ])
