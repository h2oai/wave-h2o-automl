import os
from h2o_wave import site, ui, app, Q, main
import mlops
import driverlessai
from .config import *
from .mlops_utils import *
from .utils import *

async def init_app(q: Q):
    q.page['banner'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      icon=app_config.icon, icon_color=app_config.icon_color)
    await q.page.save()
    # Get project list from MLOps Storage via MLOps Gateway + Python Client
    q.user.mlops_client = mlops.Client(
        gateway_url=os.environ['STORAGE_URL'], token_provider=lambda: q.auth.access_token)
    

    q.page['main'] = ui.form_card(
        box=app_config.main_box,
        items=[
            ui.text(f'User: {q.auth.username}'),
            #ui.text(f'Projects: {projects.project}'),
            ui.textbox(name='dai_address', label='DAI address', required=True, value='https://steam.wave.h2o.ai/proxy/driverless/5/'),
            ui.textbox(name='dai_uname', label='DAI username', required=False),
            ui.textbox(name='dai_passwd', label='DAI password', password=True, required=False),
            ui.button(name='next_dai_address', label='Next', primary=True)
        ]
    )
    await q.page.save()


async def init_dai_client(q: Q):
    if q.args.dai_uname:
        q.user.dai_client = driverlessai.Client(
            address=q.args.dai_address,
            username=q.args.dai_uname,
            password=q.args.dai_passwd
        )
    else:
        q.user.dai_client = driverlessai.Client(
            address=q.args.dai_address,
            token_provider=lambda: q.auth.access_token
        )
    await show_tables(q)


async def show_tables(q: Q):
    await clean_cards(q)
    show_dai_experiments(q, 'experiments', app_config.plot11_box)
    mlops_list_projects(q, q.user.mlops_client, 'projects', app_config.plot12_box)
    await q.page.save()


async def clean_cards(q: Q):
    cards_to_clean = ['main', 'experiments', 'projects']
    for card in cards_to_clean:
        del q.page[card]


async def show_mlops_details(q: Q):
    projects_df = q.user.projects_df
    selected_index = int(q.args.mlops_projects_table[0])
    q.user.project_id = projects_df['id'][selected_index]
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[ui.text(f'selected id {q.user.project_id}'),
                            ui.buttons([ui.button(name='next_delete_project', label='Delete Project', primary=True),
                                        ui.button(name='back_mlops_details', label='Back', primary=True)])
                            ])
    await q.page.save()
    #mlops_list_deployments(q, q.user.mlops_client, q.user.project_id)

@app('/')
async def serve(q: Q):
    # Initialiaze mlops client
    if not q.app.initialized:
        await init_app(q)
        q.app.initialized = True
    # User clicks on next on DAI selection screen
    if q.args.next_dai_address:
        await show_progress(q, 'main', app_config.main_box, f'Connecting to DAI')
        await clean_cards(q)
        await init_dai_client(q)
    elif q.args.mlops_projects_table:
        await clean_cards(q)
        await show_mlops_details(q)
    # If a user clicks on experiment in DAI table, link to project and then deploy
    elif q.args.dai_experiments_table:
        await clean_cards(q)
        await link_experiment_to_project(q)
        q.page['main'] = ui.form_card(box=app_config.main_box,
                                      items=[ui.message_bar('success', f'Imported experiment {q.user.experiment_key}'),
                                             ui.progress('Deploying in MLOps')])
        await q.page.save()
        await mlops_deploy(q)
        q.page['main'].items = [ui.message_bar('success', f'Exported experiment {q.user.experiment_key} to storage'),
                                ui.text(f'MLOps Deployment Id: **{q.user.deployment_id}**')
                                ]
        await q.page.save()
        await q.sleep(2)
        await show_tables(q)
    elif q.args.next_delete_project:
        mlops_delete_project(q, q.user.project_id)
        q.page['main'].items = [ui.message_bar('success', f'Deleted project {q.user.project_id}')]
        await q.page.save()
        await q.sleep(2)
        await show_tables(q)
    elif q.args.back_mlops_details:
        await show_tables(q)
        # MLOps predict
        #prediction = await q.run(mlops_scorer_setup, q)
        #q.page['main'].items = [ui.text(f'{q.user.sample_url} : {q.user.score_url} {prediction}')]
        #await q.page.save()

    await q.page.save()
