import os
from h2o_wave import site, ui, app, Q, main, on, handle_on
from .config import *
from .utils import *
import concurrent.futures
import asyncio
import re
import base64
from sklearn.metrics import confusion_matrix, average_precision_score, recall_score, \
    f1_score, roc_auc_score,  precision_recall_curve, auc, roc_curve
from plotly import io as pio
import plotly.express as px
import smart_open


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

    if q.app.custom_scorer:
        global_nav = [
            ui.nav_group('Navigation', items=[
                ui.nav_item(name='#home', label='Main Menu'),
                ui.nav_item(name='#import', label='Score a CSV'),
                ui.nav_item(name='#dashboard', label='Dashboard'),
            ])]
    else:
        global_nav = [ui.nav_group('Navigation', items=[])]

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
                ui.zone('body', direction=ui.ZoneDirection.ROW, size='900px'),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title=app_config.title,
                                      subtitle=app_config.subtitle , nav=global_nav)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')


# Initialize app
async def init_app_scorer(q: Q):
    q.app.app_icon, = await q.site.upload(['./static/icon.png'])
    global_nav = [
        ui.nav_group('Navigation', items=[
            ui.nav_item(name='#home', label='Main Menu'),
            ui.nav_item(name='#import', label='Score a CSV'),
            ui.nav_item(name='#dashboard', label='Dashboard'),
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
                #ui.zone('body', direction=ui.ZoneDirection.ROW, size='900px'),
                ui.zone('main', zones=[
                    # Main page single card
                    ui.zone('body'),
                    # Main page split into cards shown in vertical orientation
                    ui.zone('body_charts', direction=ui.ZoneDirection.ROW, zones=[
                        ui.zone('body_cm', size='600px'),
                        ui.zone('body_table'),
                    ]),
                   ui.zone('body_plots', direction=ui.ZoneDirection.ROW, size='600px')

                ]),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title=q.user.app_name,
                                      subtitle='', nav=global_nav)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')


@on('next_start')
@on('back_start')
async def next_start(q: Q):
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import MOJO', icon='Database'),
                ui.step(label='Step 2: Import Files', icon='Settings'),
                ui.step(label='Step 3: Settings', icon='DietPlanNotebook')]),
            ui.file_upload(name='uploaded_mojo', label='Upload MOJO', multiple=False)
        ]
    )


@on('uploaded_mojo')
@on('back_files')
async def import_files_settings(q: Q, warning: str = ''):
    if q.args.uploaded_mojo:
        q.user.mojo_file_path = q.args.uploaded_mojo[0]
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import MOJO', icon='Database', done=True),
                ui.step(label='Step 2: Import Files', icon='Settings'),
                ui.step(label='Step 3: Settings', icon='DietPlanNotebook')]),
            ui.separator('S3 Data'),
            ui.message_bar('warning', warning),
            ui.text_xl('Import Data from S3'),
            ui.textbox(name='s3_data', label='Test File (S3 URI)',
                       value=''),
            ui.buttons([ui.button(name='next_settings', label='Next', primary=True),
                        ui.button(name='back_start', label='Back', primary=False)]),
            ui.separator('Local Data'),
            ui.file_upload(name='uploaded_file', label='Upload File', multiple=True),
            ])
    await q.page.save()


@on('#import')
async def score_files_settings(q: Q, warning: str = ''):
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.text_xl('Score a CSV'),
            ui.message_bar('warning', warning),
            ui.separator('S3 Data'),
            ui.text_xl('Import Data from S3'),
            ui.textbox(name='s3_data', label='Test File (S3 URI)',
                       value=''),
            ui.buttons([ui.button(name='next_prepare_scorer', label='Next', primary=True),
                        ui.button(name='#home', label='Main Menu', primary=False)]),
            ui.separator('Local Data'),
            ui.file_upload(name='uploaded_file_score', label='Upload File', multiple=True),
            ]
    )

    await q.page.save()


@on('next_prepare_scorer')
@on('uploaded_file_score')
async def next_prepare_score(q: Q):
    if q.args.uploaded_file:
        q.user.test_path = q.args.uploaded_file[0]
    elif q.args.s3_data:
        q.user.test_df = pd.read_csv(smart_open.smart_open(q.args.s3_data))
        q.user.test_path = q.app.tmp_path+'/test.csv'
        q.user.test_df.to_csv(q.user.test_path)
    else:
        await score_files_settings(q, warning='Please select a test file')
        return
    score_result = await score_csv(q)
    if score_result:
        q.page['main'] = ui.form_card(box='body', items=[
            ui.text_xl(f'Score a CSV'),
            ui.message_bar('success', f'File scored!'),
            ui.button(name='#dashboard', label='View Dashboard', primary=True)
        ])


@on('uploaded_file')
@on('next_settings')
async def next_settings(q: Q, warning: str = ''):
    if q.args.uploaded_file:
        q.user.test_path = q.args.uploaded_file[0]
        # Download test data
        q.user.test_path = await q.site.download(q.user.test_path, q.app.tmp_path)
        q.user.test_df = pd.read_csv(q.user.test_path)
    elif q.args.s3_data:
        q.user.test_df = pd.read_csv(smart_open.smart_open(q.args.s3_data))
        q.user.test_path = q.app.tmp_path+'/test.csv'
        q.user.test_df.to_csv(q.user.test_path)
    else:
        await import_files_settings(q, 'Please select a test file')
        return

    #columns = list(q.user.test_df)
    #column_choices = [ui.choice(i, i) for i in columns]
    q.page['main'] = ui.form_card(
        box='body',
        items=[
            ui.stepper(name='icon-stepper', items=[
                ui.step(label='Step 1: Import MOJO', icon='Database', done=True),
                ui.step(label='Step 2: Import Files', icon='Settings', done=True),
                ui.step(label='Step 3: Settings', icon='DietPlanNotebook')]),
            ui.message_bar('danger', warning),
            #ui.dropdown(name='target_column', label='Target Column', choices=column_choices, required=True),
            ui.textbox(name='app_name', label='App name', required=True),
            ui.buttons([ui.button(name='next_done', label='Next', primary=True),
                        ui.button(name='back_files', label='Back', primary=False)
                        ])
        ])



@on()
async def next_done(q: Q):
    q.user.app_name = q.args.app_name
    #q.user.target_column = q.args.target_column
    mojo_name = q.user.mojo_file_path.split('/')[-1]
    # Unzip mojo to tmp folder
    filename = q.user.app_name.replace(' ', '_')
    q.user.mojo_path = await q.site.download(q.user.mojo_file_path, q.app.tmp_path)
    os.system(f'unzip -o {q.app.tmp_path}/{mojo_name} -d {q.app.tmp_path}/mojo_{filename}')
    q.user.mojo_scorer_dir = f'{q.app.tmp_path}/mojo_{filename}/mojo-pipeline'

    # FIXME uncomment before cloud
    #os.environ["DRIVERLESS_AI_LICENSE_KEY"] = ""

    # Score
    score_result = await score_csv(q)
    if score_result:
        await init_app_scorer(q)
        q.app.custom_scorer = True
        await q.page.save()

        q.page['main'] = ui.form_card(box='body', items=[
            ui.text_xl(f'MOJO to Wave Scorer'),
            ui.message_bar('success', f'Application {q.user.app_name} created successfully. Use navigation menu on top left.'),
            ui.button(name='#dashboard', label='View Dashboard', primary=True)
        ])
    await q.page.save()


@on('')
async def score_csv(q: Q):
    cmd = f'java -Xmx5g -cp {q.user.mojo_scorer_dir}/mojo2-runtime.jar ai.h2o.mojos.ExecuteMojo {q.user.mojo_scorer_dir}/pipeline.mojo {q.user.test_path} {q.app.tmp_path}/pred.csv'
    try:
        os.system(cmd)
        q.user.pred_df = pd.read_csv(f'{q.app.tmp_path}/pred.csv')
        q.user.target_column = list(q.user.pred_df.columns)[0].split('.')[0]
        q.user.results_df = pd.concat([q.user.test_df, q.user.pred_df], axis=1)
        return True
    except Exception as e:
        q.page['main'] = ui.form_card(box='body', items=[
            ui.text_xl(f'MOJO to Wave Scorer'),
            ui.message_bar('error', f'Error: {e}'),
            ui.text(f'Command run: {cmd}'),
            ui.button(name='#home', label='Main Menu', primary=True)
        ])
        return False


def get_auc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    config = {
        'scrollZoom': False,
        'showLink': False,
        'displayModeBar': False
    }
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    html = pio.to_html(fig, validate=False, include_plotlyjs='cdn', config=config)
    return html


def get_pr_curve(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    config = {
        'scrollZoom': False,
        'showLink': False,
        'displayModeBar': False
    }
    html = pio.to_html(fig, validate=False, include_plotlyjs='cdn', config=config)
    return html


@on('#dashboard')
async def show_dashboard(q: Q):

    # Confusion matrix template. Taken from: https://h2oai.github.io/wave/blog/ml-release-0.3.0
    template ="""
## Confusion Matrix
|              |           |              |
| :-:          |:-:        |:-:           |
|              | Actual 1  | Actual 0 |
| Predicted 1     | **{tp}**  | {fp} (FP)    |
| Predicted 0 | {fn} (FN) | **{tn}**     |
"""

    target_column = q.user.target_column
    y_true = q.user.results_df[target_column].to_list()
    prediction = q.user.pred_df.values.tolist()
    # Get a threshold value if available or 0.5 by default
    threshold = q.args.slider if 'slider' in q.args else 0.5

    y_pred_orig = [p[1] for p in prediction]

    # Compute confusion matrix
    y_pred = [p[1] < threshold for p in prediction]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = round(average_precision_score(y_true, y_pred)*100, 2)
    recall = round(recall_score(y_true, y_pred)*100, 2)
    f1 = round(f1_score(y_true, y_pred), 2)
    auc_score = round(roc_auc_score(y_true, y_pred), 2)

    # Handle interaction
    q.page['cm'] = ui.form_card(box='body_cm', items=[
        ui.text(template.format(tn=tn, fp=fp, fn=fn, tp=tp)),
        ui.slider(name='slider', label='Threshold', min=0, max=1, step=0.01, value=q.args.slider, trigger=True),
        ui.stats(items=[
            ui.stat(label='Precision', value=f'{precision}%'),
            ui.stat(label='Recall', value=f'{recall}%'),
            ui.stat(label='F1 Score', value=f'{f1}'),
            ui.stat(label='AUC Score', value=f'{auc_score}'),
        ], justify='between', inset=True),
    ])

    auc_html = get_auc_curve(y_true, y_pred_orig)
    q.page['plot1'] = ui.frame_card(box='body_plots', title='', content=auc_html)

    pr_html = get_pr_curve(y_true, y_pred_orig)
    q.page['plot2'] = ui.frame_card(box='body_plots', title='', content=pr_html)


    # Table
    table = table_from_df(q.user.results_df, 'results_table', searchable=True, downloadable=True, groupable=True, height='400px')
    q.page['main'] = ui.form_card(
        box='body_table',
        items=[
            ui.text_xl('Predictions Table'),
            table,
            ui.buttons([
                ui.button(name='#home', label='Main Menu', primary=True),
                #ui.button(name='back_files', label='Back', primary=False)
            ])
        ]
    )

    await q.page.save()

def main_menu(q: Q):
    q.page['main'] = ui.form_card(box='body', items=app_config.items_guide_tab)


# Clean cards before next route
async def clean_cards(q: Q):
    cards_to_clean = ['main', 'cm', 'plot1', 'plot2']
    for card in cards_to_clean:
        del q.page[card]


# Main loop
@app('/')
async def serve(q: Q):
    cur_dir = os.getcwd()
    q.app.tmp_path = cur_dir + app_config.tmp_dir
    if not os.path.exists(q.app.tmp_path):
        os.mkdir(q.app.tmp_path)

    await clean_cards(q)

    hash = q.args['#']

    # Init meta data based on state
    if q.app.custom_scorer and hash != 'home':
        await init_app_scorer(q)
    else:
        await init_app(q)

    if q.args.slider and not hash:
        await show_dashboard(q)
    elif not await handle_on(q):
        main_menu(q)
    await q.page.save()
