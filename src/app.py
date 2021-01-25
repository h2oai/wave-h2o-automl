import os
from h2o_wave import site, ui, app, Q, main
import mlops
import driverlessai
from .config import *
from .mlops_utils import *
from .utils import *
import requests
import h2osteam
from h2osteam.clients import DriverlessClient


def init_steam_client(q: Q):
    h2osteam.login(url='https://steam.wave.h2o.ai/', access_token=lambda: q.auth.access_token)
    

def init_mlops_client(q: Q):
    # Get project list from MLOps Storage via MLOps Gateway + Python Client
    q.user.mlops_client = mlops.Client(
        gateway_url=os.environ['STORAGE_URL'], token_provider=lambda: q.auth.access_token)
    print(f'Auth access token: {q.auth.access_token}')


async def init_dai_client(q: Q):
    if q.args.dai_address:
        q.user.dai_address = q.args.dai_address
    q.user.dai_client = driverlessai.Client(
            address=q.user.dai_address,
            token_provider=lambda: q.auth.access_token)
    await show_tables(q)


async def init_app(q: Q, warning: str=''):
    init_mlops_client(q)
    q.page['banner'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      icon=app_config.icon, icon_color=app_config.icon_color)
    q.page['navmenu'] = ui.toolbar_card(
        box=app_config.navbar_box,
        items=[ui.command(name="#home", label="Home", caption="Home", icon="Home"),
               ui.command(name="#steam", label="Steam", caption="Steam", icon="ConnectVirtualMachine"),
               #ui.command(name="#dai", label="DAI", caption="DAI", icon="BullseyeTarget"),
               ui.command(name="#mlops", label="MLOps", caption="MLOps", icon="OfflineStorageSolid")
               ]
    )
    q.page['main'] = ui.form_card(
        box=app_config.main_box,
        items=[
            ui.text(f'User: {q.auth.username}'),
            ui.message_bar('warning', warning),
            ui.textbox(name='dai_address', label='DAI URL from Steam', placeholder='https://steam.wave.h2o.ai/proxy/driverless/35/', required=True, value=q.user.dai_address),
            ui.button(name='next_dai_address', label='Next', primary=True)
        ]
    )
    await q.page.save()


async def show_tables(q: Q):
    await clean_cards(q)
    q.user.experiments_df = show_dai_experiments(q, 'experiments', app_config.plot11_box)
    q.user.projects = mlops_list_projects(q, q.user.mlops_client, 'projects', app_config.plot12_box)
    await q.page.save()


async def clean_cards(q: Q):
    cards_to_clean = ['main', 'experiments', 'projects']
    for card in cards_to_clean:
        del q.page[card]


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
                                             [ui.button(name='next_delete_project', label='Delete Project', primary=True),
                                              ui.button(name='back', label='Back', primary=True)])
                            ])
    q.user.deployments_df = mlops_list_deployments(q, q.user.mlops_client, q.user.project_id, 'experiments', app_config.plot2_box)
    await q.page.save()


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
                                             [ui.button(name='next_deploy_experiment', label='Create Project & Deploy Experiment', primary=True),
                                              ui.button(name='back', label='Back', primary=True)])
                                         ])


async def show_deployment_details(q: Q):
    if q.args.mlops_deployments_table:
        q.app.selected_table_index = q.args.mlops_deployments_table[0]
    deployment_index = q.app.selected_table_index

    q.user.deployment_id = q.user.deployments_df.loc[deployment_index, :]['deployment_id']
    q.user.experiment_id = q.user.deployments_df.loc[deployment_index, :]['experiment_id']
    q.user.project_id = q.user.deployments_df.loc[deployment_index, :]['project_id']
    deployment_state = q.user.deployments_df.loc[deployment_index, :]['state']
    scorer_url = q.user.deployments_df.loc[deployment_index, :]['scorer']
    sample_url = q.user.deployments_df.loc[deployment_index, :]['sample_url']
    grafana_url = q.user.deployments_df.loc[deployment_index, :]['grafana_endpoint']

    widgets = [ui.text_xl('Deployment'),
               ui.text(f'**Deployment Id:** {q.user.deployment_id}'),
               ui.text(f'**Deployment Status:** {deployment_state}'),
               ui.text(f'**Project Id:** {q.user.project_id}'),
               ui.text(f'**Experiment Id:** {q.user.experiment_id}'),
               ui.text(f'**Scorer:** {scorer_url}'),
               ui.link(label='Sample URL', path=sample_url, target='', button=True),
               ui.link(label='Grafana URL', path=grafana_url, target='', button=True),
               ]

    # Get sample query from sample URL
    if q.args.score_request:
        await show_progress(q, 'main', app_config.main_box, 'Scoring..')
        response = mlops_get_score(scorer_url, q.args.score_request)
        if response:
            q.user.scores_df = pd.DataFrame(columns=response['fields'], data=response['score'])
            score_table = table_from_df(q.user.scores_df, 'scores_table')
        else:
            score_table = ui.text('Scorer not ready. Please retry')
        widgets.extend([ui.textbox(name='score_request', label='Request Query', value=q.args.score_request),
                        ui.buttons([ui.button(name='next_score', label='Score', primary=True),
                                    ui.button(name='back', label='Back', primary=True)]),
                        ui.text_xl('Predictions'),
                        score_table])
        q.page['main'] = ui.form_card(box=app_config.main_box, items=widgets)

    else:
        await show_progress(q, 'main', app_config.main_box, 'Checking deployment status..')
        sample_query = mlops_get_sample_request(sample_url)
        widgets.extend([ui.textbox(name='score_request', label='Request Query',
                                                        value=sample_query),
                                             ui.buttons(
                                                 [ui.button(name='next_score', label='Score', primary=True),
                                                  ui.button(name='back', label='Back', primary=True)])])
        q.page['main'] = ui.form_card(box=app_config.main_box, items=widgets)


async def deploy_experiment(q: Q):
    # Link experiment to project
    await link_experiment_to_project(q)
    q.page['main'] = ui.form_card(box=app_config.main_box,
                                  items=[ui.message_bar('success', f'Imported experiment {q.user.experiment_key}'),
                                         ui.progress('Deploying in MLOps')])
    await q.page.save()

    # Deploy experiment
    mlops_client = q.user.mlops_client
    project_key = q.user.project_key
    experiment_key = q.user.experiment_key
    deployment = mlops_deploy_experiment(
        mlops_client=mlops_client,
        project_id=project_key,
        experiment_id=experiment_key,
        type=mlops.StorageDeploymentType.SINGLE_MODEL
    )
    q.user.deployment_id = deployment.id
    q.page['main'].items = [ui.message_bar('success', f'Exported experiment {q.user.experiment_key} to storage'),
                            ui.text(f'MLOps Deployment Id: **{q.user.deployment_id}**')
                            ]
    await q.page.save()
    await q.sleep(2)
    await show_tables(q)


@app('/')
async def serve(q: Q):
    hash = q.args['#']
    if hash == 'home':
        await clean_cards(q)
        await init_app(q)
    elif hash == 'steam':
        await clean_cards(q)
        q.page['main'] = ui.frame_card(box=app_config.main_box, title='Steam', path='https://steam.wave.h2o.ai/')
    elif hash == 'dai':
        await clean_cards(q)
        if q.user.dai_address:
            q.page['main'] = ui.frame_card(box=app_config.main_box, title='DAI', path=q.user.dai_address)
        else:
            q.page['main'] = ui.form_card(box=app_config.main_box, items=[ui.text('Please enter DAI credentials in Home menu')])
    elif hash == 'mlops':
        await clean_cards(q)
        q.page['main'] = ui.frame_card(box=app_config.main_box, title='MLOps', path='https://mlops.wave.h2o.ai/')

    # Initialiaze mlops client
    elif not q.app.initialized:
        await init_app(q)
        q.app.initialized = True
    # User clicks on next on DAI selection screen
    elif q.args.next_dai_address:
        if not q.args.dai_address:
            await init_app(q, 'Please enter DAI address in Steam. eg: https://steam.wave.h2o.ai/proxy/driverless/5/')
            return
        await show_progress(q, 'main', app_config.main_box, f'Connecting to DAI')
        await init_dai_client(q)
    # User clicks on a project in the MLOps table
    elif q.args.mlops_projects_table:
        await clean_cards(q)
        init_mlops_client(q)
        await show_mlops_details(q)
    # If a user clicks on experiment in DAI table, link to project and then deploy
    elif q.args.dai_experiments_table:
        await clean_cards(q)
        await show_experiment_details(q)
    # User selects experiment deployment
    elif q.args.next_deploy_experiment:
        await clean_cards(q)
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
        await clean_cards(q)
        await show_deployment_details(q)
    # User scoring a request using a deployment
    elif q.args.next_score:
        await clean_cards(q)
        await show_deployment_details(q)
    elif q.args.back:
        await clean_cards(q)
        await init_dai_client(q)
    else:
        await clean_cards(q)
        await init_app(q)
    await q.page.save()
