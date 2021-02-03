import os
from h2o_wave import site, ui, app, Q, main
import mlops
import driverlessai
from .config import *
from .mlops_utils import *
from .utils import *
from .steam_utils import *
import concurrent.futures
import asyncio
import h2osteam

LOCAL_TEST = False
if LOCAL_TEST:
    STEAM_URL = 'https://steam.wave.h2o.ai/'
else:
    STEAM_URL = os.environ['STEAM_URL']


# Initialize Steam client
def init_steam_client(q: Q):
    if LOCAL_TEST:
        token=''
        h2osteam.login(url=STEAM_URL, username='trushant.kalyanpur@h2o.ai', password=token)
    else:
        h2osteam.login(url=STEAM_URL, username=q.auth.username, access_token=q.auth.access_token,
                       verify_ssl=not STEAM_URL.startswith("http://"))


# Initialize MLOPs client
def init_mlops_client(q: Q):
    # Get project list from MLOps Storage via MLOps Gateway + Python Client
    q.user.mlops_client = mlops.Client(
        gateway_url=os.environ['STORAGE_URL'], token_provider=lambda: q.auth.access_token)
    print(f'Auth access token: {q.auth.access_token}')


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


# Main landing page with description
def landing_page(q: Q):
    q.page['main'] = ui.form_card(box=app_config.main_box, items=app_config.items_guide_tab)


# Initialize app
def init_app(q: Q, warning: str = ''):
    global_nav = [
        ui.nav_group('Navigation', items=[
            ui.nav_item(name='#home', label='Home'),
            ui.nav_item(name='#steam', label='Steam'),
            ui.nav_item(name='#dai', label='DAI'),
            ui.nav_item(name='#mlops', label='MLOPs'),
        ])]

    if LOCAL_TEST:
        init_steam_client(q)
    else:
        init_steam_client(q)
        init_mlops_client(q)
    q.page['banner'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      nav=global_nav)
    landing_page(q)


# Show DAI experiments and MLOP's Projects
async def show_tables(q: Q):
    await clean_cards(q)
    q.user.experiments_df = show_dai_experiments(q, 'experiments', app_config.plot11_box)
    q.user.projects = mlops_list_projects(q, q.user.mlops_client, 'projects', app_config.plot12_box)
    await q.page.save()


# Clean cards before next route
async def clean_cards(q: Q):
    cards_to_clean = ['main', 'experiments', 'projects']
    for card in cards_to_clean:
        del q.page[card]


# Show MLOPs details
async def show_mlops_details(q: Q):
    projects = q.user.projects
    selected_index = int(q.args.mlops_projects_table[0])
    q.user.project_id = projects[selected_index].id
    q.user.project_name = projects[selected_index].display_name
    q.page['main'] = ui.form_card(box=app_config.plot1_box,
                                  items=[ui.text_xl('MLOPs Project'),
                                         ui.text(f'**MLOPs Project Name:**: {q.user.project_name}'),
                                         ui.text(f'**MLOPs Project Id:**: {q.user.project_id}'),
                                         ui.buttons(
                                             [ui.button(name='next_delete_project', label='Delete Project',
                                                        primary=True),
                                              ui.button(name='back', label='Back', primary=True)])
                                         ])
    status, code = mlops_list_deployments(q, q.user.mlops_client, q.user.project_id, 'experiments',
                                          app_config.plot2_box)
    if status:
        q.user.deployments_df = code
    else:
        show_error(q, 'Model is still being deployed. Please try again in a minute', app_config.main_box)
        time.sleep(2)
        show_tables(q)
    await q.page.save()


# Show DAI experiment details
async def show_experiment_details(q: Q):
    experiment_index = int(q.args.dai_experiments_table[0])
    q.user.experiment_key = q.user.experiments_df['Experiment Key'][experiment_index]
    q.page['main'] = ui.form_card(box=app_config.main_box,
                                  items=[ui.text_xl('DAI Experiment'),
                                         ui.text(f'**Experiment Key:** {q.user.experiment_key}'),
                                         ui.text(f'**Experiment Name:** {q.user.experiment_name}'),
                                         ui.textbox(name='project_name', label='Project Name'),
                                         ui.textbox(name='project_desc', label='Project Description'),
                                         ui.buttons(
                                             [ui.button(name='next_deploy_experiment',
                                                        label='Create Project & Deploy Experiment', primary=True),
                                              ui.button(name='back', label='Back', primary=True)])
                                         ])


# Show user selected deployment
async def show_deployment_details(q: Q):
    if q.args.mlops_deployments_table:
        q.app.selected_table_index = q.args.mlops_deployments_table[0]
    deployment_index = q.app.selected_table_index

    q.user.deployment_id = q.user.deployments_df.loc[deployment_index, :]['deployment_id']
    q.user.experiment_id = q.user.deployments_df.loc[deployment_index, :]['experiment_id']
    q.user.project_id = q.user.deployments_df.loc[deployment_index, :]['project_id']
    deployment_state = q.user.deployments_df.loc[deployment_index, :]['state']
    q.user.scorer_url = q.user.deployments_df.loc[deployment_index, :]['scorer']
    sample_url = q.user.deployments_df.loc[deployment_index, :]['sample_url']
    grafana_url = q.user.deployments_df.loc[deployment_index, :]['grafana_endpoint']

    # Score using request
    if q.args.score_request:
        request_str = q.args.score_request
        await show_progress(q, 'main', app_config.main_box, 'Scoring..')
        response = mlops_get_score(q.user.scorer_url, q.args.score_request)
        if response:
            q.user.scores_df = pd.DataFrame(columns=response['fields'], data=response['score'])
            score_table = table_from_df(q.user.scores_df, 'scores_table', downloadable=True)
        else:
            score_table = ui.text('Scorer not ready. Please retry')
        r_widgets = [ui.text_xl('Predictions'), score_table]
    # Score using CSV
    elif q.user.scoring_via_df:
        request_str = q.user.sample_query
        results_table = table_from_df(q.user.results_df, 'results_table', downloadable=True)
        r_widgets = [ui.text_xl('Predictions'), results_table]
    # Default
    else:
        request_str = q.user.sample_query
        await show_progress(q, 'main', app_config.main_box, 'Checking deployment status..')
        q.user.sample_query = mlops_get_sample_request(sample_url)
        sample_query_dict = json.loads(q.user.sample_query)
        q.user.sample_df = pd.DataFrame(data=sample_query_dict['rows'], columns=sample_query_dict['fields'])
        q.user.sample_df.to_csv('sample_scoring.csv', index=False)
        q.user.sample_download_path, = await q.site.upload(['sample_scoring.csv'])
        r_widgets = []

    widgets = [ui.text_xl('Deployment'),
               ui.text(f'**Deployment Id:** {q.user.deployment_id}'),
               ui.text(f'**Deployment Status:** {deployment_state}'),
               ui.text(f'**Project Id:** {q.user.project_id}'),
               ui.text(f'**Experiment Id:** {q.user.experiment_id}'),
               # ui.text(f'**Scorer:** "{q.user.scorer_url}"'),
               ui.inline([
                   ui.link(label='Sample URL', path=sample_url, target='', button=True),
                   ui.link(label='Grafana URL', path=grafana_url, target='', button=True)]),
               ui.button(name='back', label='Back', primary=True),
               ui.separator('Score a CSV'),
               ui.text(f'[Download sample CSV to score]({q.user.sample_download_path})'),
               ui.button(name='next_score_df', label='Score CSV', primary=True),
               ui.separator('Score a Request'),
               ui.textbox(name='score_request', label='Request Query', value=request_str),
               ui.button(name='next_score', label='Score Request', primary=True)
               ]

    q.page['main'] = ui.form_card(box=app_config.plot11_box, items=widgets)
    q.page['projects'] = ui.form_card(box=app_config.plot12_box, items=r_widgets)


# Deploy DAI experiment to MLOPs
async def deploy_experiment(q: Q):
    # Link experiment to project
    status, code = await link_experiment_to_project(q)
    # Deploy experiment if export was a success
    if status:
        q.page['main'] = ui.form_card(box=app_config.main_box,
                                      items=[ui.message_bar('success', f'Imported experiment {q.user.experiment_key}'),
                                             ui.progress('Deploying in MLOps'),
                                             ui.button(name='next_tables', label='Next', primary=True)
                                             ])
        # Deploy experiment
        mlops_client = q.user.mlops_client
        project_key = q.user.project_key
        experiment_key = q.user.experiment_key
        try:
            deployment = mlops_deploy_experiment(
                mlops_client=mlops_client,
                project_id=project_key,
                experiment_id=experiment_key,
                type=mlops.StorageDeploymentType.SINGLE_MODEL
            )
            q.user.deployment_id = deployment.id
            q.page['main'].items = [
                ui.message_bar('success', f'Exported experiment {q.user.experiment_key} to storage'),
                ui.text(f'MLOps Deployment Id: **{q.user.deployment_id}**'),
                ui.button(name='next_tables', label='Next', primary=True)
            ]
        except Exception as e:
            q.page['main'].items = [ui.message_bar('warning', f'Error: {e}'),
                                    ui.button(name='next_tables', label='Next', primary=True)
                                    ]
    else:
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.message_bar('warning', f'Error: {code}'),
            ui.button(name='next_tables', label='Next', primary=True)
        ])
        # Delete project now because of the error
        mlops_delete_project(q, q.user.project_key)
    await q.page.save()


# Menu for importing new datasets
async def import_menu(q: Q, card_name, card_box):
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('Import Data'),
        ui.file_upload(name='uploaded_file', label='Upload File', multiple=True),
    ])


# Main loop
@app('/')
async def serve(q: Q):
    hash = q.args['#']
    await clean_cards(q)
    if hash == 'home':
        init_app(q)
    elif hash == 'steam' or q.args.next_steam_menu:
        #q.page['main'] = ui.frame_card(box=app_config.main_box, title='Steam', path='https://steam.wave.h2o.ai/')
        steam_menu(q)
    elif hash == 'dai':
        if q.user.dai_address:
            q.page['main'] = ui.frame_card(box=app_config.main_box, title='DAI', path=q.user.dai_address_auth)
        else:
            q.page['main'] = ui.form_card(box=app_config.main_box,
                                          items=[ui.text('No DAI instance found. Please connect via Steam first.'),
                                                 ui.button(name='next_steam_menu', label='Next', primary=True)
                                                 ])
    elif hash == 'mlops':
        #q.page['main'] = ui.frame_card(box=app_config.main_box, title='MLOps', path='https://mlops.wave.h2o.ai/')
        await init_dai_client(q)
    elif q.args.next_steam_menu or q.args.refresh_steam_menu:
        steam_menu(q)
    # Initialiaze mlops client
    elif not q.app.initialized:
        init_app(q)
        q.app.initialized = True
    # User clicks on next on DAI selection screen
    elif q.args.next_connect_instance:
        await show_progress(q, 'main', app_config.main_box, f'Connecting to DAI')
        await init_dai_client(q)
    # User clicks on a project in the MLOps table
    elif q.args.mlops_projects_table:
        init_mlops_client(q)
        await show_mlops_details(q)
    # If a user clicks on experiment in DAI table, link to project and then deploy
    elif q.args.dai_experiments_table:
        await show_experiment_details(q)
    # User selects experiment deployment
    elif q.args.next_deploy_experiment:
        await deploy_experiment(q)
    # User deletes a project
    elif q.args.next_delete_project:
        mlops_delete_project(q, q.user.project_id)
        q.page['main'] = ui.form_card(box=app_config.main_box,
                                      items=[ui.message_bar('success', f'Deleted project {q.user.project_id}')])
        await q.page.save()
        await q.sleep(2)
        await show_tables(q)
    # User clicks on a row in the deployments table
    elif q.args.mlops_deployments_table:
        await show_deployment_details(q)
    # User scoring a request using a deployment
    elif q.args.next_score:
        q.user.scoring_via_df = False
        await show_deployment_details(q)
    elif q.args.back:
        await init_dai_client(q)
    elif q.args.next_score_df:
        await import_menu(q, 'main', app_config.main_box)
        # User uploads a file
    elif q.args.uploaded_file:
        uploaded_file_path = q.args.uploaded_file
        for file in uploaded_file_path:
            filename = file.split('/')[-1]
            uploaded_files_dict[filename] = uploaded_file_path
        local_path = await q.site.download(uploaded_file_path[0], '.')
        time.sleep(1)
        scoring_df = pd.read_csv(local_path)
        q.user.results_df = get_predictions_df(q.user.scorer_url, scoring_df)
        q.user.scoring_via_df = True
        await show_deployment_details(q)
    elif q.args.next_tables:
        await show_tables(q)
    elif q.args.next_create_instance and q.args.instance_name:
        await create_dai_instance(q)
    elif q.args.steam_table:
        steam_selection(q)
    elif q.args.next_instance_start_stop:
        await start_stop_instance(q)
    elif q.args.next_delete_instance:
        await terminate_instance(q)
    else:
        q.user.scoring_via_df = False
        init_app(q)
    await q.page.save()
