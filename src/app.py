import h2o
from h2o.automl import H2OAutoML
from h2o_wave import Q, ui, app, main, data, copy_expando, on, handle_on
from .config import *
import time
import pandas as pd
import asyncio
import concurrent.futures
import numpy as np
import base64
import io
from loguru import logger

h2o.init()
app_config = Configuration()


# Initialize app
def init_app(q: Q):
    if not q.client.icon_color:
        q.client.icon_color = '#CDDD38'

    q.page['meta'] = ui.meta_card(box='',title='H2O AutoML', layouts=[
        ui.layout(
            breakpoint='300px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('nav', direction=ui.ZoneDirection.ROW, size='90px', zones=[
                    ui.zone(name='tabs'),
                    ui.zone(name='misc'),
                ]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('charts', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('charts_left', direction=ui.ZoneDirection.COLUMN),
                            ui.zone('charts_right', direction=ui.ZoneDirection.COLUMN),
                        ])
                    ]),
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='600px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('nav', direction=ui.ZoneDirection.ROW, size='90px', zones=[
                    ui.zone(name='tabs'),
                    ui.zone(name='misc'),
                ]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('charts', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('charts_left', direction=ui.ZoneDirection.COLUMN, size='300px'),
                            ui.zone('charts_right', direction=ui.ZoneDirection.COLUMN, size='300px'),
                        ])
                    ]),
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='1000px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('nav', direction=ui.ZoneDirection.ROW, size='90px', zones=[
                    ui.zone(name='tabs'),
                    ui.zone(name='misc'),
                ]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('charts', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('charts_left', direction=ui.ZoneDirection.COLUMN, size='500px'),
                            ui.zone('charts_right', direction=ui.ZoneDirection.COLUMN, size='500px'),
                        ])
                    ]),
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='1400px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('nav', direction=ui.ZoneDirection.ROW, size='90px', zones=[
                    ui.zone(name='tabs'),
                    ui.zone(name='misc'),
                ]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('charts', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('charts_left', direction=ui.ZoneDirection.COLUMN, size='700px'),
                            ui.zone('charts_right', direction=ui.ZoneDirection.COLUMN, size='700px'),
                        ])
                    ]),
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='1700px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('nav', direction=ui.ZoneDirection.ROW, size='90px', zones=[
                    ui.zone(name='tabs'),
                    ui.zone(name='misc'),
                ]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('charts', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('charts_left', direction=ui.ZoneDirection.COLUMN, size='850px'),
                            ui.zone('charts_right', direction=ui.ZoneDirection.COLUMN, size='850px'),
                        ])
                        ]),
                ]),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title=app_config.title, subtitle=app_config.subtitle,
                                      icon='Settings', icon_color=q.client.icon_color)
    q.page['footer'] = ui.footer_card(box='footer', caption='Made with 💛 using <a href="https://wave.h2o.ai" target="_blank">H2O Wave</a>')


async def update_theme(q: Q):
    """
    Update theme of app.
    """

    copy_expando(q.args, q.client)

    if q.client.theme_dark:
        q.client.icon_color = 'black'
        q.page['meta'].theme = 'neon'
        q.page['header'].icon_color = q.client.icon_color
        q.client.img_source = 'https://i.imgur.com/yH2zRAm.jpg'
    else:
        q.client.icon_color = '#CDDD38'
        q.page['meta'].theme = 'light'
        q.page['header'].icon_color = q.client.icon_color
        q.client.img_source = 'https://i.imgur.com/jLrt5mr.png'

    q.page['misc'].items[2].toggle.value = q.client.theme_dark

    q.page['main'].items[0].text.content = f'<center><img width="240" height=240" src="{q.client.img_source}"></center>'

    await q.page.save()


@on('guide')
async def main_menu(q: Q):
    q.app.df_rows = []
    if not q.client.img_source:
        q.client.img_source = 'https://i.imgur.com/jLrt5mr.png'

    q.page['menu'] = ui.tab_card(
        box='tabs',
        items=[
            ui.tab(name="guide", label="Home",  icon="Home"),
            ui.tab(name="import", label="Import Data",  icon="Database"),
            ui.tab(name="train", label="Train",  icon="BullseyeTarget"),
            ui.tab(name="lb", label="Leaderboard", icon="ClipboardList"),
            ui.tab(name="explain", label="AutoML Viz", icon="ClipboardList"),
            ui.tab(name="explain2", label="Model Explain", icon="ClipboardList"),
        ],
        link=True
    )

    # Logo
    if not q.app.logo_url:
        q.app.logo_url, = await q.site.upload([logo_file])

    q.page['misc'] = ui.section_card(
        box='misc',
        title='',
        subtitle='',
        items=[
            ui.link(label='Documentation', path='https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html', target=''),
            ui.text(content=''),
            ui.toggle(name='theme_dark', label='Dark Mode', value=q.client.theme_dark, trigger=True)
        ]
    )

    q.page['main'] = ui.form_card(box='body_main', items=[
            ui.text(f"""
<center><img width="240" height=240" src="{q.client.img_source}"></center>"""),
            #ui.frame(content='<h2><center>H2O-3 AutoML</center></h2>', height='60px'),
            ui.text_xl('<p style="text-align: center;">H2O-3 AutoML</p>'),
            ui.text("""
This Wave application demonstrates how to use H2O-3 AutoML via the Wave UI.
### **Features**:
* **AutoML Training**: Allows a user to train a model using H2O-3 AutoML on custom train/test datasets.<br>
* **Leaderboard**: Visualizing the AML leaderboard.<br>
* **Explainability**: Shows feature importance and row Shapley contributions. <br>
* **Deployment**: Select a model for MOJO download.<br>

Reference: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
            """),
        ])
    await q.page.save()


# Menu for importing new datasets
@on('import')
async def import_menu(q: Q):
    q.page['main'] = ui.form_card(box='body_main', items=[
        ui.text_xl('Import Data'),
        ui.file_upload(name='uploaded_file', label='Upload File', multiple=True),
    ])


async def upload_data(q: Q):
    uploaded_file_path = q.args.uploaded_file
    for file in uploaded_file_path:
        filename = file.split('/')[-1]
        uploaded_files_dict[filename] = uploaded_file_path
    time.sleep(1)
    q.page['main'] = ui.form_card(box='body_main',
                                  items=[ui.message_bar('success', 'File Imported! Please select an action'),
                                         ui.buttons([ui.button(name='train', label='Train a model', primary=True),
                                                     ui.button(name='guide', label='Main Menu', primary=False)])])


# Menu for selecting a pre-loaded table
@on('train')
async def select_table(q: Q, arg=False, warning: str = ''):
    choices = []
    if uploaded_files_dict:
        for file in uploaded_files_dict:
            choices.append(ui.choice(file, file))
        q.page['main'] = ui.form_card(box= ui.box('body_main', width='500px'), items=[
            ui.message_bar(type='warning', text=warning),
            ui.dropdown(name='train_file', label='Training data', value=q.app.train_file, required=True,
                        choices=choices),
            ui.dropdown(name='test_file', label='Test data (optional)', value=q.args.test_file, required=False,
                        choices=choices),
            ui.buttons([ui.button(name='selected_tables_next', label='Next', primary=True)])
        ])
    else:
        q.page['main'] = ui.form_card(box=ui.box('body_main', width='500px'), items=[
            ui.text_xl(f'{q.app.task}'),
            ui.message_bar(type='warning', text=warning),
            ui.text(f'No data found. Please import data first.'),
            ui.buttons([ui.button(name='import', label='Import Data', primary=True)])
        ])


# Settings for training
async def train_menu(q: Q, warning: str = ''):
    # Error handling
    if not q.args.train_file and not q.app.train_file:
        await select_table(q, False, 'Please select training data')
        return




    # Store train/test file
    if q.args.train_file:
        q.app.train_file = q.args.train_file
        train_path = uploaded_files_dict[q.app.train_file][0]
        # Check for data provided as part of app
        if 'data/credit_card_train.csv' or 'data/wine_quality_train.csv' in train_path:
            local_path = train_path
        else:
            local_path = await q.site.download(train_path, '.')
        q.app.train_df = pd.read_csv(local_path)
    if q.args.test_file:
        q.app.test_file = q.args.test_file
        test_path = uploaded_files_dict[q.app.test_file][0]
        # Check for data provided as part of app
        if 'data/credit_card_test.csv' or 'data/wine_quality_test.csv' in test_path:
            local_path = test_path
        else:
            local_path = await q.site.download(test_path, '.')
        q.app.test_df = pd.read_csv(local_path)
    # Default options
    if not q.app.max_models:
        q.app.max_models = 10
    if not q.app.is_classification:
        q.app.is_classification = True
    if not q.app.nfolds:
        q.app.nfolds = '5'
    if not q.app.es_metric:
        q.app.es_metric = 'AUTO'
    if not q.app.es_rounds:
        q.app.es_rounds = '3'

    es_metrics = ['AUTO', 'deviance', 'logloss', 'MSE', 'RMSE', 'MAE', 'RMSLE', 'AUC', 'AUCPR', 'lift_top_group',
                  'misclassification', 'mean_per_class_error']
    es_metrics_choices = [ui.choice(i, i) for i in es_metrics]
    choices = [ui.choice(i, i) for i in list(q.app.train_df.columns)]
    q.page['main'] = ui.form_card(box=ui.box('body_main', width='500px'), items=[
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
        ui.textbox(name='es_rounds', label='Early stopping rounds', value=str(q.app.es_rounds)),
        ui.textbox(name='nfolds', label='nfolds', value=str(q.app.nfolds)),
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
    q.page['main'] = ui.form_card(box='body_main', items=[])
    if not q.args.target:
        await train_menu(q, 'Please select target column')
        return

    q.app.target = q.args.target
    q.app.max_models = q.args.max_models
    q.app.max_runtime_mins = q.args.max_runtime_mins
    q.app.es_metric = q.args.es_metric
    q.app.es_rounds = int(q.args.es_rounds)
    q.app.nfolds = int(q.args.nfolds)
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

    # Run AutoML (limited to 1 hour max runtime by default)
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
    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()]
    table = ui.table(name=f'{table_name}',
             rows=rows,
             columns=columns,
             multiple=False,  # Allow multiple row selection
             height='400px')
    return table


# Show leaderboard
@on('lb')
@on('back_lb')
async def show_lb(q: Q):
    if q.app.aml:
        # H2O automl object
        aml = q.app.aml

        # Get leaderboard
        #lb = aml.leaderboard
        lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
        lb_df = lb.as_data_frame()

        #round data to 5 decimal places
        cols = lb_df.columns.to_list()
        cols.remove('model_id')
        cols.remove('training_time_ms')
        cols.remove('predict_time_per_row_ms')
        cols.remove('algo')
        for col in cols:
            lb_df[col] = lb_df[col].round(5)


        lb_table = table_from_df(lb_df, 'lb_table')

        q.page['main'] = ui.form_card(box='body_main', items=[
            ui.text_xl('AutoML Leaderboard'),
            ui.text(f'**Training shape:** {q.app.train_df.shape}'),
            ui.text(f'**Target:** {q.app.target}'),
            ui.text_m(f'**Select a model to get the MOJO**'),
            lb_table
        ])
    else:
        q.page['main'] = ui.form_card(box='body_main', items=[
            ui.text_xl('AutoML Leaderboard'),
            ui.text('No models trained. Please train a model first.'),
            ui.buttons([ui.button(name='train', label='Train a model', primary=True)])
        ])


# Clean cards
async def clean_cards(q: Q):
    # TO DO: update this to clean new explain cards
    cards_to_clean = ['main', 'plot1', 'plot21', 'plot22', 'plot31', 'plot32', 'foo']
    for card in cards_to_clean:
        del q.page[card]


# Image from matplotlib object
def get_image_from_matplotlib(matplotlib_obj):
    if hasattr(matplotlib_obj, "figure"):
        matplotlib_obj = matplotlib_obj.figure()
    buffer = io.BytesIO()
    matplotlib_obj.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# Get MOJO for selected row in table
@on('lb_table')
async def get_mojo(q: Q):

    await clean_cards(q)

    if q.args.lb_table:
        q.app.selected_model = q.args.lb_table[0]
    if not q.app.shap_row_index:
        q.app.shap_row_index = str(0)
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
    shap_choices = [ui.choice(str(i), str(i)) for i in range(min(10, q.app.train_df.shape[0]))]

    q.page['main'] = ui.form_card(box='body_main', items=[
        ui.text_l(f'**Model:** {model_str}'),
        ui.text(f'[Download MOJO]({download_path})'),
        ui.dropdown(name='shap_row_index', label='Select Row for Shapley Plot', value=q.app.shap_row_index,
                    choices=shap_choices, trigger=True),
        ui.buttons([ui.button(name='back_lb', label='Back to Leaderboard', primary=True)])
    ])

    # Variable importance plot
    try:
        varimp_plot = model.varimp_plot(server = True)
        q.page['plot21'] = ui.image_card(
            box='charts_left',
            title="Variable Importance Plot",
            type="png",
            image=get_image_from_matplotlib(varimp_plot),
        )
    except Exception as e:
        print(f'No variable importance found for {model_str}: {e}')
        q.page['plot21'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Variable importance unavailable for **{model_str}**')
           ])

    #try:
    #    var_imp_df = model.varimp(use_pandas=True)
    #    sorted_df = var_imp_df.sort_values(by='scaled_importance', ascending=True).iloc[0:10]
    #    rows = list(zip(sorted_df['variable'], sorted_df['scaled_importance']))
    #    q.page.add('plot21', ui.plot_card(
    #        box=ui.box('charts_left', height='300px'),
    #        title='Variable Importance Plot',
    #        data=data('feature score', rows=rows),
    #        plot=ui.plot([ui.mark(type='interval', x='=score', y='=feature', x_min=0, y_min=0, x_title='Relative Importance',
    #                              y_title='Feature', color='#33BBFF')])
    #    ))
    #except Exception as e:
    #    print(f'No var_imp found for {model_str}: {e}')
    #    q.page['plot21'] = ui.form_card(box='charts_left', items=[
    #        ui.text(f'Variable importance unavailable for **{model_str}**')
    #       ])


    # SHAP plot
    try:
        shap_plot = model.shap_explain_row_plot(frame=train, row_index=int(q.app.shap_row_index), figsize=(FIGSIZE[0], FIGSIZE[0]))
        q.page['plot22'] = ui.image_card(
            box='charts_right',
            title="Shapley Plot",
            type="png",
            image=get_image_from_matplotlib(shap_plot),
        )
    except Exception as e:
        print(f'No shap found for {model_str}: {e}')
        q.page['plot22'] = ui.form_card(box='charts_right', items=[
            ui.text(f'Shapley unavailable for **{model_str}**')
           ])
    # Learning curve plots
    # TO DO: why is this showing up on all the tabs?
    try:
        learning_curve_plot = model.learning_curve_plot(figsize=FIGSIZE)
        q.page['plot31'] = ui.image_card(
            box='charts_left',
            title="Learning curve Plot",
            type="png",
            image=get_image_from_matplotlib(learning_curve_plot),
        )
    except Exception as e:
        print(f'No Learning Curve found for {model_str}: {e}')
        q.page['plot31'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Learning Curve unavailable for **{model_str}**')
           ])







#AutoML Level Plots Tab
@on('explain')
async def aml_plots(q: Q, arg=False, warning: str = ''):

    await clean_cards(q)

    q.page['main'] = ui.tab_card(
        box='body_main',
        value = 'automl_summary',
        items=[
            ui.tab(name="automl_summary", label="Models Summary",  icon="Home"),#model correlation + pareto front (to do)
            ui.tab(name="automl_varimp", label="Variable Explain",  icon="Database"),#varimp heatmap + PD plot + picker
        ],
        link=True
    )

    # Model Correlation Heatmap (1)
    try:
        train = h2o.H2OFrame(q.app.train_df)
        y = q.app.target
        if q.app.is_classification:
            train[y] = train[y].asfactor()
        mc_plot = q.app.aml.model_correlation_heatmap(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]))
        q.page['plot21'] = ui.image_card(
            box='charts_left',
            title="Model Correlation Heatmap Plot",
            type="png",
            image=get_image_from_matplotlib(mc_plot),
        )
    except Exception as e:
        print(f'No model correlation heatmap found: {e}')
        q.page['plot21'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Model correlation heatmap unavailable')
           ])


    # Model Correlation Heatmap (2)
    # Duplicated for now, but needs to be replaced with pareto front
    #try:
    #    train = h2o.H2OFrame(q.app.train_df)
    #    y = q.app.target
    #    if q.app.is_classification:
    #        train[y] = train[y].asfactor()
    #    mc_plot = q.app.aml.model_correlation_heatmap(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]))
    #    q.page['plot22'] = ui.image_card(
    #        box='charts_right',
    #        title="Model Correlation Heatmap Plot",
    #        type="png",
    #        image=get_image_from_matplotlib(mc_plot),
    #    )
    #except Exception as e:
    #    print(f'No model correlation heatmap found: {e}')
    #    q.page['plot22'] = ui.form_card(box='charts_right', items=[
    #        ui.text(f'Model correlation heatmap unavailable')
    #       ])


# This is currently the same code as the aml_plots above
# we should use one function twice
@on('automl_summary')
async def aml_summary(q: Q, arg=False, warning: str = ''):

    await clean_cards(q)

    q.page['main'] = ui.tab_card(
        box='body_main',
        value = 'automl_summary',
        items=[
            ui.tab(name="automl_summary", label="Models Summary",  icon="Home"),#model correlation + pareto front (to do)
            ui.tab(name="automl_varimp", label="Variable Explain",  icon="Database"),#varimp heatmap + PD plot + picker
        ],
        link=True
    )

    # Model Correlation Heatmap (1)
    try:
        train = h2o.H2OFrame(q.app.train_df)
        y = q.app.target
        if q.app.is_classification:
            train[y] = train[y].asfactor()
        mc_plot = q.app.aml.model_correlation_heatmap(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]))
        q.page['plot21'] = ui.image_card(
            box='charts_left',
            title="Model Correlation Heatmap Plot",
            type="png",
            image=get_image_from_matplotlib(mc_plot),
        )
    except Exception as e:
        print(f'No model correlation heatmap found: {e}')
        q.page['plot21'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Model correlation heatmap unavailable')
           ])

    # Model Correlation Heatmap (2)
    # Duplicated for now, but needs to be replaced with pareto front
    #try:
    #    train = h2o.H2OFrame(q.app.train_df)
    #    y = q.app.target
    #    if q.app.is_classification:
    #        train[y] = train[y].asfactor()
    #    mc_plot = q.app.aml.model_correlation_heatmap(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]))
    #    q.page['plot22'] = ui.image_card(
    #        box='charts_right',
    #        title="Model Correlation Heatmap Plot",
    #        type="png",
    #        image=get_image_from_matplotlib(mc_plot),
    #    )
    #except Exception as e:
    #    print(f'No model correlation heatmap found: {e}')
    #    q.page['plot22'] = ui.form_card(box='charts_right', items=[
    #        ui.text(f'Model correlation heatmap unavailable')
    #       ])


@on('automl_varimp')
async def aml_varimp(q: Q, arg=False, warning: str = ''):

    await clean_cards(q)

    q.page['main'] = ui.tab_card(
        box='body_main',
        value = 'automl_varimp',
        items=[
            ui.tab(name="automl_summary", label="Models Summary",  icon="Home"),#model correlation + pareto front (to do)
            ui.tab(name="automl_varimp", label="Variable Explain",  icon="Database"),#varimp heatmap + PD plot + picker
        ],
        link=True
    )

    # Variable Importance Heatmap
    try:
        varimp_heat_plot = q.app.aml.varimp_heatmap()
        q.page['plot21'] = ui.image_card(
            box='charts_left',
            title="Variable Importance Heatmap Plot",
            type="png",
            image=get_image_from_matplotlib(varimp_heat_plot),
        )
    except Exception as e:
        print(f'No variable importance heatmap found: {e}')
        q.page['plot21'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Variable importance heatmap unavailable')
           ])


    #on right side now
    choices = []
    x = q.app.train_df.columns.to_list()
    x.remove(q.app.target)
    if x:
        for col in x:
            choices.append(ui.choice(col, col))

    if q.args.column_pd is not None:
        col = q.args.column_pd[0]
        q.page['plot31'] = ui.form_card(box='charts_right', items=[
            ui.picker(name='column_pd', label='Select Column', choices=choices, max_choices = 1, values = q.args.column_pd),
            ui.buttons([ui.button(name='select_column_pd', label='Show Partial Dependence', primary=True)])
        ])
    else:
        q.page['plot31'] = ui.form_card(box='charts_right', items=[
            ui.picker(name='column_pd', label='Select Column', choices=choices, max_choices = 1, values = [q.app.aml.get_best_model(algorithm="basemodel").varimp()[0][0]]),
            ui.buttons([ui.button(name='select_column_pd', label='Show Partial Dependence', primary=True)])
        ])


    # PD Plot
    try:
        train = h2o.H2OFrame(q.app.train_df)
        y = q.app.target
        if q.app.is_classification:
            train[y] = train[y].asfactor()


        if q.args.column_pd is not None:
            col = q.args.column_pd[0]
        else:
            m = q.app.aml.get_best_model(algorithm="basemodel")
            vi = m.varimp()
            col = vi[0][0]


        pd_plot = q.app.aml.pd_multi_plot(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]), column = col)
        q.page['plot32'] = ui.image_card(
            box='charts_right',
            title="Partial Dependence Multi-model Plot",
            type="png",
            image=get_image_from_matplotlib(pd_plot),
        )
    except Exception as e:
        print(f'No Partial Dependence Multi-model Plot found: {e}')
        q.page['plot32'] = ui.form_card(box='charts_right', items=[
            ui.text(f'Partial Dependence Multi-model Plot unavailable')
           ])



# Menu for importing new datasets
@on('explain2')
async def picker_example(q: Q, arg=False, warning: str = ''):
    await clean_cards(q)
    choices = []
    models_list = q.app.aml.leaderboard.as_data_frame()['model_id'].to_list()
    if models_list:
        for model_id in models_list:
            choices.append(ui.choice(model_id, model_id))
    print(choices)
    if q.args.model_picker is not None:
        q.page['main'] = ui.form_card(box='body_main', items=[
            ui.picker(name='model_picker', label='Select Model', choices=choices, max_choices = 1, values = q.args.model_picker),
            ui.buttons([ui.button(name='select_model', label='Explain Model', primary=True)])
        ])
    else:
        q.page['main'] = ui.form_card(box='body_main', items=[
            ui.picker(name='model_picker', label='Select Model', choices=choices, max_choices = 1, values = [q.app.aml.get_best_model(algorithm="basemodel").model_id]),
            ui.buttons([ui.button(name='select_model', label='Explain Model', primary=True)])
        ])
    # TO DO: actually do something when the user clicks on the button
    # - create the plots using that particular model


    if q.args.model_picker is not None:
        model_str = q.args.model_picker[0]
        model = h2o.get_model(q.args.model_picker[0])
    else:
        model = q.app.aml.get_best_model(algorithm="basemodel")
        model_str = "Model not found"

    # Variable importance plot
    try:
        varimp_plot = model.varimp_plot(server = True)
        q.page['plot21'] = ui.image_card(
            box='charts_left',
            title="Variable Importance Plot",
            type="png",
            image=get_image_from_matplotlib(varimp_plot),
        )
    except Exception as e:
        print(f'No variable importance found for {model_str}: {e}')
        q.page['plot21'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Variable importance unavailable for **{model_str}**')
           ])


    # Shapley Summary Plot
    try:
        train = h2o.H2OFrame(q.app.train_df)
        y = q.app.target
        if q.app.is_classification:
            train[y] = train[y].asfactor()

        shap_plot = model.shap_summary_plot(frame = train, figsize=(FIGSIZE[0], FIGSIZE[0]))
        q.page['plot22'] = ui.image_card(
            box='charts_right',
            title="Shapley Summary Plot",
            type="png",
            image=get_image_from_matplotlib(shap_plot),
        )
    except Exception as e:
        print(f'No shapley summary found for {model_str}: {e}')
        q.page['plot22'] = ui.form_card(box='charts_right', items=[
            ui.text(f'Shapley Summary unavailable for **{model_str}**')
           ])

    #Learning Curve Plot
    try:
        learning_curve_plot = model.learning_curve_plot(figsize=FIGSIZE)
        q.page['plot31'] = ui.image_card(
            box='charts_left',
            title="Learning curve Plot",
            type="png",
            image=get_image_from_matplotlib(learning_curve_plot),
        )
    except Exception as e:
        print(f'No Learning Curve found for {model_str}: {e}')
        q.page['plot31'] = ui.form_card(box='charts_left', items=[
            ui.text(f'Learning Curve unavailable for **{model_str}**')
           ])

# async def generate_explain_automl(q: Q):

#     # TO DO: Figure out how to populate the group/automl level plots
#     # - how to access the automl object from the other tab

#     # Variable importance plot
#     try:
#         var_imp_df = model.varimp(use_pandas=True)
#         sorted_df = var_imp_df.sort_values(by='scaled_importance', ascending=True).iloc[0:10]
#         rows = list(zip(sorted_df['variable'], sorted_df['scaled_importance']))
#         q.page.add('foo', ui.plot_card(
#             box='charts_left',
#             title='Variable Importance Plot',
#             data=data('feature score', rows=rows),
#             plot=ui.plot([ui.mark(type='interval', x='=score', y='=feature', x_min=0, y_min=0, x_title='Relative Importance',
#                                   y_title='Feature', color='#33BBFF')])
#         ))
#     except Exception as e:
#         model_str = "foo"
#         print(f'No var_imp found for {model_str}: {e}')
#         q.page['foo'] = ui.form_card(box='charts_left', items=[
#             ui.text(f'Variable importance unavailable for **{model_str}**')
#            ])


@app('/')
async def serve(q: Q):
    cur_dir = os.getcwd()
    q.app.tmp_dir = cur_dir + app_config.tmp_dir
    if not os.path.exists(q.app.tmp_dir):
        os.mkdir(q.app.tmp_dir)

    # Default is light mode
    if not q.client.theme_dark:
        q.client.theme_dark = False

    if not q.client.app_initialized:
        init_app(q)
        q.client.app_initialized = True

    # Clean cards
    await clean_cards(q)

    # Hash routes user when tabs are clicked
    if q.args.theme_dark is not None and q.args.theme_dark != q.client.theme_dark:
        await update_theme(q)
    # User selected files from import menu
    elif q.args.uploaded_file:
        await upload_data(q)
    # User selected train/test file
    elif q.args.selected_tables_next:
        await train_menu(q)
    # User starts training
    elif q.args.next_train:
        await train_model(q)
    elif q.args.shap_row_index and q.args.shap_row_index != q.app.shap_row_index:
        await get_mojo(q)
    # User clicks explain tab
    elif q.args.select_column_pd:
        await aml_varimp(q)
    elif q.args.select_model:
        await picker_example(q)
    elif not await handle_on(q):
        await main_menu(q)
    await q.page.save()
