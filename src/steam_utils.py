from h2o_wave import site, ui, app, Q, main, on
from .config import *
from .utils import *
import h2osteam
from h2osteam.clients import DriverlessClient,H2oKubernetesClient
import asyncio
import h2osteam

# Show steam menu on left and steam table on right
@on('#steam')
@on('refresh_steam')
@on('back_steam_table')
async def steam_view(q: Q):
    menu_box = app_config.steam_menu_box
    h2o_running_instances_df = q.user.h2o_instances_df[q.user.h2o_instances_df['status'] == 'running']
    h2o_unstopped_instances_df = q.user.h2o_instances_df[~q.user.h2o_instances_df['status'].isin(['running','stopped'])]
    dai_running_instances = list(q.user.dai_instances_df[q.user.dai_instances_df['status'] == 'running']['name'].values)
    h2o_running_instances = list(h2o_running_instances_df['name'].values)
    dai_choices = [ui.choice(i, i) for i in dai_running_instances]
    h2o_choices = [ui.choice(i, i) for i in h2o_running_instances]

    q.user.instances_df = pd.concat(
        [q.user.dai_instances_df, h2o_running_instances_df, h2o_unstopped_instances_df]).reset_index(drop=True)

    widgets = [
        ui.text("H2O AI Cloud Engines"),
        ui.text('<img src="http://0xdata-images.s3.amazonaws.com/h2o-driverless-ai.png" width="200px" />'),
    ]

    # DAI Menu Selection
    if dai_running_instances:
        widgets.extend([
            ui.dropdown(name='dai_name', label='DAI Instance', choices=dai_choices, required=True),
            ui.buttons([ui.button(name='next_train', label='Next', primary=True)], justify='start')
        ])
    else:
        widgets.extend([
            ui.text('No running DAI instances found'),
        ])
    widgets.extend([
        ui.separator(""),
        ui.text('<img src="https://www.h2o.ai/wp-content/uploads/2018/09/h2o_logo.svg" width="200px" />')
    ])

    # H2O Menu Selection
    if h2o_running_instances:
        widgets.extend([
            ui.dropdown(name='h2o_name', label='H2O-3 Instance', choices=h2o_choices, required=True),
            ui.buttons([ui.button(name='next_train', label='Next', primary=True)], justify='start')
        ])
    else:
        widgets.extend([
            ui.text('No running H2O instances found.'),
        ])

    # Allow user to proceed if both DAI and H2O running instances found
    #if dai_running_instances and h2o_running_instances:
    #    widgets.extend([
    #        ui.buttons([ui.button(name='next_train', label='Next', primary=True)], justify='start')
    #    ])

    q.page['steam_menu'] = ui.form_card(box=menu_box, items=widgets)
    await show_steam_table(q)

# Get users DAI and H2O instances and store in df
def get_steam_instances(q: Q):
    # Find dai and h2o3 instances of user
    dai_instances = h2osteam.api().get_driverless_instances()
    if dai_instances:
        dai_instances_df = pd.DataFrame(dai_instances)
        dai_instances_df = dai_instances_df[['id', 'name', 'status', 'version']]
    else:
        dai_instances_df = pd.DataFrame(columns=['id', 'name', 'status', 'version'])
    q.user.dai_instances_df = dai_instances_df

    h2o_instances = H2oKubernetesClient.get_clusters()
    if h2o_instances:
        h2o_names = []
        h2o_status = []
        h2o_ids = []
        h2o_versions = []
        for inst in h2o_instances:
            h2o_names.append(inst.name)
            h2o_ids.append(inst.id)
            h2o_status.append(inst.status)
            h2o_versions.append(inst.version)
        h2o_instances_df = pd.DataFrame(
            {'id': h2o_ids, 'name': h2o_names, 'status': h2o_status, 'version': h2o_versions})
    else:
        h2o_instances_df = pd.DataFrame(columns=['id', 'name', 'status'])
    q.user.h2o_instances_df = h2o_instances_df
    

# Show users DAI instances and running H2O instances
async def show_steam_table(q: Q):
    box = app_config.steam_table_box
    table = table_from_df(q.user.instances_df, 'steam_table', searchable=False, groupable=False)
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_l('H2O AI Cloud - Steam Instances'),
        ui.text_m('Click on a row to start/stop an instance'),
        table,
        ui.buttons([
            ui.button(name='launch_dai_menu', label='Launch new DAI instance', primary=True),
            ui.button(name='launch_h2o_menu', label='Launch new H2O instance', primary=True),
            ui.button(name='refresh_steam', label='Refresh', primary=False),
        ])
    ])
    await q.page.save()


@on('launch_dai_menu')
# Launch menu for a new DAI instance
async def launch_dai_menu(q: Q):
    box = app_config.steam_table_box
    dai_choices = [ui.choice(i, i) for i in ['1.9.1.1']]
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_xl('DAI Instance'),
        ui.dropdown(name='instance_version', label='DAI Version', choices=dai_choices, value='1.9.0.6'),
        ui.textbox(name='instance_name', label='Instance name', required=True),
        ui.textbox(name='instance_cpu_count', label="Number of CPU's", value='1'),
        ui.textbox(name='instance_mem', label='MEMORY [GB]', value='4'),
        ui.textbox(name='instance_storage', label='STORAGE [GB]', value='10'),
        ui.textbox(name='instance_idle_time', label='MAXIMUM IDLE TIME [HRS]', value='12'),
        ui.textbox(name='instance_uptime', label='MAXIMUM UPTIME [HRS]', value='8'),
        ui.textbox(name='instance_timeout', label='TIMEOUT [S]', value='600'),
        ui.buttons([
            ui.button(name='launch_dai', label='Next', primary=True),
            ui.button(name='back_steam_table', label='Back', primary=False)
            ])
    ])


# Launch menu for a new H2O instance
@on('launch_h2o_menu')
async def launch_h2o_menu(q: Q):
    box = app_config.steam_table_box
    h2o_vers = ['3.32.0.5']
    h2o_ver_choices = [ui.choice(i,i) for i in h2o_vers]
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_l('H2O Instance'),
        ui.textbox(name='h2o_cluster_name', label='H2O Cluster name'),
        ui.dropdown(name='h2o_version', label='H2O Version', choices=h2o_ver_choices),
        ui.textbox(name='node_count', label='Number of nodes', value='4'),
        ui.textbox(name='cpu_count', label='Number of CPUS', value='1'),
        ui.textbox(name='memory_gb', label='Number of nodes', value='1'),
        ui.textbox(name='max_idle_h', label='Number of nodes', value='8'),
        ui.textbox(name='max_uptime_h', label='Number of nodes', value='12'),
        ui.textbox(name='timeout_s', label='Number of nodes', value='600'),
        ui.buttons([
            ui.button(name='launch_h2o', label='Launch new H2O cluster', primary=True),
            ui.button(name='back_steam_table', label='Back', primary=False)
        ])
    ])


# Progress after user launches H2O
@on('launch_h2o')
async def launch_h2o(q: Q):
    box = app_config.steam_table_box
    name = q.args.h2o_cluster_name
    version = q.args.h2o_version
    node_count = int(q.args.node_count)
    cpu_count = int(q.args.cpu_count)
    memory_gb = int(q.args.memory_gb)
    max_idle_h = int(q.args.max_idle_h)
    max_uptime_h = int(q.args.max_idle_h)
    timeout_s = int(q.args.timeout_s)
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_xl('H2O Cluster'),
        ui.progress(f'Starting H2O cluster {name}')
    ])

    await q.page.save()
    H2oKubernetesClient.launch_cluster(name=name,
                                       version=version,
                                       node_count=node_count,
                                       cpu_count=cpu_count,
                                       gpu_count=0,
                                       memory_gb=memory_gb,
                                       max_idle_h=max_idle_h,
                                       max_uptime_h=max_uptime_h,
                                       timeout_s=timeout_s)

    del q.page['steam']
    await steam_view(q)


# Progress after user launches DAI
@on('launch_dai')
async def launch_dai(q: Q):
    box = app_config.steam_table_box
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_xl('DAI Instance'),
        ui.progress(f'Starting DAI instance {q.args.instance_name}')
    ])

    await q.page.save()
    DriverlessClient.launch_instance(name=q.args.instance_name,
                                         version=q.args.instance_version,
                                         profile_name="default-driverless-kubernetes",
                                         gpu_count=0,
                                         memory_gb=int(q.args.instance_mem),
                                         storage_gb=int(q.args.instance_storage),
                                         max_idle_h=int(q.args.instance_idle_time),
                                         max_uptime_h=int(q.args.instance_uptime),
                                         timeout_s=int(q.args.instance_timeout)
                                         )
    del q.page['steam']
    await steam_view(q)


# Shown when a user clicks on a row in the steam table
@on('steam_table')
async def steam_selection(q: Q):
    box = app_config.steam_table_box
    selected_id = int(q.args.steam_table[0])
    instances_df = q.user.instances_df

    # Store user selected instance
    instance_id = instances_df.iloc[selected_id, :]['id']
    q.user.instance_id = instance_id
    #q.user.dai_address = f'https://steam.cloud.h2o.ai/proxy/driverless/{instance_id}/'
    #q.user.dai_address_auth = f'https://steam.cloud.h2o.ai/oidc-login-start?forward=/proxy/driverless/{q.user.instance_id}/openid/callback'
    instance_name = instances_df.iloc[selected_id, :]['name']
    instance_status = instances_df.iloc[selected_id, :]['status']
    instance_version = instances_df.iloc[selected_id, :]['version']
    q.user.instance_name = instance_name
    q.user.instance_status = instance_status
    q.user.instance_version = instance_version

    # Show instance options
    q.page['steam'] = ui.form_card(box=box, items=[
        ui.text_xl('Instance'),
        ui.inline([
            ui.text(f"**Name:** {instance_name}"),
            ui.text(f"**Status:** {instance_status}"),
            ui.text(f"**Version:** {instance_version}")
        ]
        ),
        ui.buttons([
            ui.button(name='next_instance_start_stop', label='Start/Stop Instance', primary=True),
            #ui.button(name='next_delete_instance', label='Delete Instance', primary=False),
            ui.button(name='back_steam_table', label='Back', primary=False),
        ])
    ])


async def start_stop_instance(instance, start_stop_str):
    if start_stop_str == 'start':
        instance.start()
    else:
        instance.stop()


# Handle user start/stop of instance
@on('next_instance_start_stop')
async def steam_start_stop_instance(q: Q):
    box = app_config.steam_table_box
    instance_name = q.user.instance_name
    instance_status = q.user.instance_status
    instance_version = q.user.instance_version
    dai = False
    # DAI choice
    if '1.9' in instance_version:
        instance = DriverlessClient.get_instance(name=instance_name)
        dai = True
    else:
        instance = H2oKubernetesClient.get_cluster(name=instance_name)

    # Start instance if stopped
    if instance_status == 'stopped':
        q.page['steam'] = ui.form_card(box=box, items=[
            ui.text_xl('Steam Instances'),
            ui.progress('Instance is starting. Please wait', caption='This may take a few minutes')
        ])
        await q.page.save()
        if dai:
            await start_stop_instance(instance, 'start')
            del q.page['steam']
            await steam_view(q)
    # Stop instance if started
    elif instance_status == 'running':
        q.page['steam'] = ui.form_card(box=box, items=[
            ui.text_xl('Steam Instances'),
            ui.progress('Instance is stopping. Please wait', caption='This may take a few minutes')
        ])
        await q.page.save()
        await start_stop_instance(instance, 'stop')
        del q.page['steam']
        await steam_view(q)


# FIXME not implemented yet
async def terminate_instance(q: Q):
    await show_progress(q, 'main', app_config.main_box, f'Terminating DAI Instance {q.user.instance_name}. Please wait..')
    instance = DriverlessClient.get_instance(name=q.user.instance_name)
    if q.user.instance_status != 'stopped':
        instance.stop()
    instance.terminate()
    steam_menu(q)
    await q.page.save()


