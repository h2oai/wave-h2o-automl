import os
from h2o_wave import site, ui, app, Q, main, on, handle_on
from .config import *
from .utils import *
from .steam_utils import *
import concurrent.futures
import asyncio
import h2osteam
import h2o

LOCAL_TEST = True
if LOCAL_TEST:
    STEAM_URL = 'https://steam.wave.h2o.ai/'
    # Get this token from steam on cloud
    username = ''
    token = ''
    h2osteam.login(url=STEAM_URL, username=username, password=token)
else:
    STEAM_URL = os.environ['STEAM_URL']
    h2osteam.login(url=STEAM_URL, username=q.auth.username, access_token=q.auth.access_token,
                   verify_ssl=not STEAM_URL.startswith("http://"))


# Initialize app
def init_app(q: Q):
    global_nav = [
        ui.nav_group('Navigation', items=[
            ui.nav_item(name='#home', label='Home'),
            ui.nav_item(name='#steam', label='Steam'),
        ])]

    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='xs',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW, zones=[ui.zone('header_title', size='30%'),
                                                                         ui.zone('header_nav', size='70%')]),
                ui.zone('body', direction=ui.ZoneDirection.COLUMN, zones=[
                    ui.zone('body_menu', size='400px'),
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('body_charts', direction=ui.ZoneDirection.COLUMN),
                        # Main page split into 2 columns, with right column have have vertical orientation
                        ui.zone('body_table_text_wc', direction=ui.ZoneDirection.COLUMN, size='400px'),
                        ui.zone('body_table_charts', size='500px')

                    ])
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='m',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW, zones=[ui.zone('header_title', size='30%'),
                                                                         ui.zone('header_nav', size='70%')]),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('body_menu', size='400px'),
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main'),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('body_charts', direction=ui.ZoneDirection.COLUMN),
                        # Main page split into 2 columns, with right column have have vertical orientation
                        ui.zone('body_table_text_wc', direction=ui.ZoneDirection.COLUMN, size='400px'),
                        ui.zone('body_table_charts', size='500px')

                    ])
                ]),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='xl',
            width='1700px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('main', zones=[
                        # Main page single card
                        ui.zone('body_main', direction=ui.ZoneDirection.ROW),
                        # Main page split into cards shown in vertical orientation
                        ui.zone('body_charts', direction=ui.ZoneDirection.COLUMN),
                        # Main page split into 2 columns, with right column have have vertical orientation
                        ui.zone('body_table', direction=ui.ZoneDirection.COLUMN, zones=[
                            ui.zone('body_table_text_wc', direction=ui.ZoneDirection.ROW, size='400px'),
                            ui.zone('body_table_charts', size='500px')
                        ]),
                    ])
                ]),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title=app_config.title, subtitle=app_config.subtitle, nav=global_nav)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')


def main_menu(q: Q):
    q.page['main'] = ui.form_card(box='body_main', items=app_config.items_guide_tab)


# Clean cards before next route
async def clean_cards(q: Q):
    cards_to_clean = ['main', 'steam', 'steam_menu']
    for card in cards_to_clean:
        del q.page[card]

@on('#home')
async def home_view(q: Q):
    main_menu(q)



@on('next_train')
async def downstream_task(q: Q):
    q.user.dai_name = q.args.dai_name
    q.user.h2o_name = q.args.h2o_name

    # Get DAI client handle
    instance = DriverlessClient.get_instance(name=q.user.dai_name)
    q.user.h2oai = instance.connect()

    # Get H2O Client handle
    #cluster = H2oKubernetesClient.get_cluster(q.user.h2o_name)
    #h2o.connect(config=cluster.get_connection_config())

    user_dai_instances = q.user.dai_instances_df
    user_h2o_instances = q.user.h2o_instances_df

    print('Done')
    
    

# Main loop
@app('/')
async def serve(q: Q):
    init_app(q)
    # Stores users DAI and H2O instances in q.user.dai_instances_df and q.user.h2o_instances_df
    get_steam_instances(q)
    await clean_cards(q)
    if not await handle_on(q):
        main_menu(q)
    await q.page.save()
