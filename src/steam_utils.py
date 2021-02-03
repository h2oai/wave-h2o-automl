from h2o_wave import site, ui, app, Q, main
from .config import *
from .utils import *
import h2osteam
from h2osteam.clients import DriverlessClient
import asyncio


def steam_menu(q: Q, warning: str = ''):
    instances = h2osteam.api().get_driverless_instances()
    if instances:
        instances_df = pd.DataFrame(instances)
        instances_df = instances_df[['id', 'name', 'status']]
    else:
        instances_df = pd.DataFrame(columns=['id', 'name', 'status'])
    q.user.instances_df = instances_df

    table = table_from_df(instances_df, 'steam_table')
    q.page['experiments'] = ui.form_card(box=app_config.plot11_box, items=[
        ui.text_xl('Steam - Existing DAI Instances'),
        ui.text_m('Select an instance from the rows'),
        ui.button(name='refresh_steam_menu', label='Refresh', primary=True),
        table
    ])

    dai_choices = [ui.choice(i, i) for i in ['1.9.0.4', '1.9.0.6']]
    q.page['main'] = ui.form_card(box=app_config.plot12_box, items=[
        ui.text_xl('Steam - New DAI Instance'),
        ui.dropdown(name='instance_version', label='DAI Version', choices=dai_choices, value='1.9.0.6'),
        ui.textbox(name='instance_name', label='Instance name', required=True),
        ui.textbox(name='instance_cpu_count', label="Number of CPU's", value=1),
        ui.textbox(name='instance_mem', label='MEMORY [GB]', value=4),
        ui.textbox(name='instance_storage', label='STORAGE [GB]', value=10),
        ui.textbox(name='instance_idle_time', label='MAXIMUM IDLE TIME [HRS]', value=12),
        ui.textbox(name='instance_uptime', label='MAXIMUM UPTIME [HRS]', value=8),
        ui.textbox(name='instance_timeout', label='TIMEOUT [S]', value=600),
        ui.button(name='next_create_instance', label='Next', primary=True)
    ])


def steam_selection(q: Q):
    selected_id = int(q.args.steam_table[0])
    instances_df = q.user.instances_df

    # Store user selected instance
    instance_id = instances_df.iloc[selected_id,:]['id']
    q.user.instance_id = instance_id
    q.user.dai_address = f'https://steam.wave.h2o.ai/proxy/driverless/{instance_id}/'
    q.user.dai_address_auth = f'https://steam.wave.h2o.ai/oidc-login-start?forward=/proxy/driverless/{q.user.instance_id}/openid/callback'
    instance_name = instances_df.iloc[selected_id,:]['name']
    q.user.instance_name = instance_name
    instance_status = instances_df.iloc[selected_id,:]['status']
    q.user.instance_status = instance_status

    # Show instance options
    q.page['main'] = ui.form_card(box=app_config.main_box, items = [
        ui.text_xl('DAI Instance'),
        ui.inline([ui.text(f"**Name:** {instance_name}"),
                   ui.text(f"**Status:** {instance_status}")]
                   ),
        ui.button(name='#mlops', label='Connect to Instance', primary=True),
        ui.buttons([
            ui.button(name='next_instance_start_stop', label='Start/Stop Instance', primary=False),
            ui.button(name='next_delete_instance', label='Delete Instance', primary=False),
            ui.button(name='back', label='Back', primary=False),
        ])
    ])


async def trigger_instance(q: Q):
    instance = DriverlessClient.get_instance(name=q.user.instance_name)
    if q.user.instance_status == 'stopped':
        await show_progress(q, 'main', app_config.main_box, f'Starting DAI Instance {q.user.instance_name}. Please wait..')
        instance.start()
    else:
        await show_progress(q, 'main', app_config.main_box, f'Stopping DAI Instance {q.user.instance_name}. Please wait..')
        instance.stop()


async def start_stop_instance(q: Q):

    q.app.future = asyncio.ensure_future(trigger_instance(q))
    # if q.user.instance_status == 'stopped':
    #     await show_progress(q, 'main', app_config.main_box, f'Starting DAI Instance {q.user.instance_name}')
    # else:
    #     await show_progress(q, 'main', app_config.main_box, f'Stopping DAI Instance {q.user.instance_name}')
    #with concurrent.futures.ThreadPoolExecutor() as pool:
    #    await q.exec(pool, trigger_instance, q)
    await q.sleep(1)
    q.app.future.cancel()
    steam_menu(q)
    await q.page.save()


async def terminate_instance(q: Q):
    await show_progress(q, 'main', app_config.main_box, f'Terminating DAI Instance {q.user.instance_name}. Please wait..')
    instance = DriverlessClient.get_instance(name=q.user.instance_name)
    if q.user.instance_status != 'stopped':
        instance.stop()
    instance.terminate()
    steam_menu(q)
    await q.page.save()


async def create_dai_instance(q: Q):
    try:
        await show_progress(q, 'main', app_config.main_box, f'Creating DAI instance {q.args.instance_name}')
        DriverlessClient.launch_instance(name=q.args.instance_name,
                                         version=q.args.instance_version,
                                         profile_name="default-driverless-kubernetes",
                                         gpu_count=0,
                                         memory_gb=q.args.instance_mem,
                                         storage_gb=q.args.instance_storage,
                                         max_idle_h=q.args.instance_idle_time,
                                         max_uptime_h=q.args.instance_uptime,
                                         timeout_s=q.args.instance_timeout
                                         )
        widgets = [ui.message_bar('success', f'Created DAI instance {q.args.instance_name}'),
                   ui.button(name='next_steam_menu', label='Next', primary=True)
                   ]
        q.page['main'] = ui.form_card(box=app_config.main_box, items=widgets)
        await q.page.save()
    except Exception as e:
        show_error(q, e, app_config.main_box)
        await q.page.save()
