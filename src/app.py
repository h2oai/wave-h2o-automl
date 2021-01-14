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
            ui.textbox(name='dai_address', label='DAI address', required=True),
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
    await show_dai_experiments(q, 'main', app_config.main_box)


@app('/')
async def serve(q: Q):
    # Initialiaze mlops client
    if not q.app.initialized:
        await init_app(q)
    # User clicks on next on DAI selection screen
    if q.args.next_dai_address:
        await show_progress(q, 'main', app_config.main_box, f'Connecting to DAI')
        await init_dai_client(q)
    # If a user clicks on experiment in DAI table, link to project and then deploy
    elif q.args.dai_experiments_table:
        await link_experiment_to_project(q)
        q.page['main'].items = [ui.message_bar('success', f'Imported experiment {q.user.experiment_key}'),
                                ui.progress('Deploying in MLOps')]
        await q.page.save()

        # MLOps deploy
        #await show_success(q, 'main', app_config.main_box, f'Imported experiment {q.user.experiment_key}')
        await mlops_deploy(q)
        projects = q.user.mlops_client.storage.project.list_projects(mlops.StorageListProjectsRequest())

        q.page['main'].items = [ui.message_bar('success', f'Exported experiment {q.user.experiment_key} to storage'),
                                ui.text(f'MLOps Deployment Id: **{q.user.deployment_id}**'),
                                ui.text(f'MLOps Projects: {projects}')
                                ]
        await q.page.save()

        # MLOps predict
        #prediction = await q.run(mlops_scorer_setup, q)
        #q.page['main'].items = [ui.text(f'{q.user.sample_url} : {q.user.score_url} {prediction}')]
        #await q.page.save()

    await q.page.save()
