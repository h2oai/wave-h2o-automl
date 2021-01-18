import os
from h2o_wave import site, ui, app, Q, main
from .config import *
from .simulators.event_sim import *
import asyncio
import concurrent.futures


# Assign app to variable
def assign_col(arg_col, store_col):
    if arg_col:
        store_col = arg_col
    return store_col


def assign_setting_vars(q: Q):
    q.app.settings_tab = assign_col(q.args.settings_tab, q.app.settings_tab)
    q.app.bed_capacity = assign_col(q.args.bed_capacity, q.app.bed_capacity)
    q.app.simulation_time = assign_col(q.args.simulation_time, q.app.simulation_time)
    q.app.arrival_interval = assign_col(q.args.arrival_interval, q.app.arrival_interval)
    q.app.incoming_population_size = assign_col(q.args.incoming_population_size, q.app.incoming_population_size)
    q.app.population_size = assign_col(q.args.population_size, q.app.population_size)
    q.app.age_range = assign_col(q.args.age_range, q.app.age_range)
    q.app.conditions = assign_col(q.args.conditions, q.app.conditions)
    q.app.dai_license = assign_col(q.args.dai_license, q.app.dai_license)


async def settings_menu(q: Q, warning: str=''):
    if not q.app.settings_tab:
        q.app.settings_tab = 'simulation'

    assign_setting_vars(q)

    if not q.app.age_range:
        q.app.age_range = [18, 80]
    if not q.app.conditions:
        q.app.conditions = ['Injuries']
    if not q.app.arrival_interval:
        q.app.arrival_interval = [1, 4]
    if not q.app.bed_capacity:
        q.app.bed_capacity = 100
    if not q.app.simulation_time:
        q.app.simulation_time = 20
    if not q.app.population_size:
        q.app.population_size = 50

    # Header for app
    q.page['header'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      icon=app_config.icon, icon_color=app_config.icon_color)

    # Tabs for settings menu
    tabs = [ui.tab(name='simulation', label='Simulation'),
            ui.tab(name='patient', label='Patient'),
            ui.tab(name='dai', label='DAI Settings')]

    patient_conditions =['Allergic-Rhinitis','Allergies','Appendicitis','Asthma', 'Atopy',
                         'Attention-Deficit-Disorder', 'Bronchitis',
                         'Colorectal-Cancer' ,'Contraceptives', 'Contraceptive-Maintenance', 'Copd',
                         'Dementia', 'Dermatitis' , 'Ear-Infections', 'Epilepsy' , 'Female-Reproduction',
                         'Fibromyalgia', 'Food-Allergies' ,'Gout', 'Homelessness', 'Injuries',
                         'Lung-Cancer','Lupus','Med-Rec', 'Metabolic-Syndrome-Care', 'Metabolic-Syndrome-Disease',
                         'Opioid-Addiction', 'Osteoarthritis', 'Osteoporosis', 'Pregnancy', 'Rheumatoid-Arthritis',
                         'Self-Harm' ,'Sexual-Activity', 'Sinusitis', 'Sore-Throat', 'Total-Joint-Replacement',
                         'Urinary-Tract-Infections','Wellness-Encounters']

    condition_choices = [ui.choice(i, i) for i in patient_conditions]

    if q.app.settings_tab == 'simulation':
        q.page['menu'] = ui.form_card(box=app_config.menu_box, items=[ui.tabs(name='settings_tab',
                                                                               value=q.app.settings_tab, items=tabs),
                                                                       ui.text_xl('Simulation Settings'),
                                                                       ui.slider(name='bed_capacity',
                                                                                 label='Bed Capacity',
                                                                                 min=10, max=2000, value=q.app.bed_capacity),
                                                                       ui.slider(name='simulation_time',
                                                                                 label='Simulation Time (days)',
                                                                                 min=10, max=365, value=q.app.simulation_time),
                                                                       ui.range_slider(name='arrival_interval',
                                                                                       label='Patient Arrival Interval (days)',
                                                                                       min=1, max=20,
                                                                                       min_value=q.app.arrival_interval[0],
                                                                                       max_value=q.app.arrival_interval[1]),
                                                                       ui.slider(name='incoming_population_size',
                                                                                 label='Incoming Population Size (per day)',
                                                                                 min=1, max=50, value=q.app.incoming_population_size),
                                                                       ui.slider(name='population_size',
                                                                                 label='Total Population Size',
                                                                                 min=2, max=500, value=q.app.population_size),
                                                                       ui.button(name='next', label='Next',
                                                                                 primary=True)
                                                                       ])
    elif q.app.settings_tab == 'patient':
        q.page['menu'] = ui.form_card(box=app_config.menu_box, items=[ui.tabs(name='settings_tab',
                                                                               value=q.app.settings_tab, items=tabs),
                                                                       ui.text_xl('Patient Settings'),
                                                                       ui.range_slider(name='age_range',
                                                                                       label='Age range',
                                                                                       min=18, max=90,
                                                                                       min_value=q.app.age_range[0],
                                                                                       max_value=q.app.age_range[1]),
                                                                       ui.dropdown(name='conditions',
                                                                                   label='Condition Modules',
                                                                                   choices=condition_choices,
                                                                                   values=q.app.conditions),
                                                                       ui.button(name='next', label='Next',
                                                                                 primary=True)
                                                                       ])
    elif q.app.settings_tab == 'dai':
        q.page['menu'] = ui.form_card(box=app_config.menu_box, items=[ui.tabs(name='settings_tab',
                                                                               value=q.app.settings_tab, items=tabs),
                                                                      ui.text_xl('DAI Settings'),
                                                                      ui.message_bar('warning', warning),
                                                                      ui.textbox(name='dai_license', label='DAI License',
                                                                                 password=True, required=True,
                                                                                 value=q.app.dai_license),
                                                                      ui.button(name='next', label='Next',
                                                                                primary=True)
                                                                      ])

    q.page['main'] = ui.form_card(box=app_config.small_main_box, items=app_config.items_guide_tab)
    await q.page.save()

async def clean_cards(q: Q):
    cards_to_clean = ['menu', 'main', 'plot1', 'plot21', 'plot22', 'button_bar', 'notification',
                      'plot01', 'plot02', 'plot03', 'plot04', 'plot05', 'plot06', 'plot07', 'plot08']
    for card in cards_to_clean:
        del q.page[card]
    await q.page.save()


@app('/')
async def serve(q: Q):
    # Initialiaze app
    #if not q.args.start_sim:
     #   await clean_cards(q)
      #  await settings_menu(q)
    # Logo
    if not q.app.logo_url:
        q.app.logo_url, = await q.site.upload([app_config.logo_file])
        q.app.app_icon_url, = await q.site.upload([app_config.app_icon_file])

    q.page['logo'] = ui.markup_card(
         box=app_config.logo_box,
         title='',
         content="""<p style='text-align:center; vertical-align: middle; display: table-cell; width: 134px;'>"""
                """<a href='https://www.h2o.ai/h2o-q/'> <img src='""" + q.app.logo_url + """' height='50px' width='50px'> </a> </p>"""

         )

    if q.args.next:
        if (not q.args.dai_license and not q.app.dai_license):
            q.app.settings_tab = 'dai'
            await settings_menu(q, 'Please enter DAI license')
            return
        assign_setting_vars(q)
        await clean_cards(q)
        await setup_mojo(q)
        await setup_data(q)
    elif q.args.start_sim:
        await clean_cards(q)
        q.app.future = asyncio.ensure_future(run_loop(q))
    elif q.args.restart_sim:
        await clean_cards(q)
        q.app.stats_plotted = False
        q.app.limit_reached = False
        await settings_menu(q)
    else:
        await clean_cards(q)
        await settings_menu(q)
    await q.page.save()
