import numpy as np
import pandas as pd
from h2o_wave import main, app, Q, ui, data
from ..config import *
from ..utils import *
from .synthea_config import *
from .synthea_utils import *
pd.options.mode.chained_assignment = None

test_mode = 0

# Global variables
class GlobalVars:
    # Track all patients processed
    patient_count = 0
    # Track waiting patient count
    wait_count = 0
    # Dict of patient class
    all_patients = {}
    bed_capacity = 0
    bed_count = 0
    min_beds = 0
    forecasted_beds = 0
    # List of time ticks
    time_list = []
    # List of beds available over time
    bed_avail_list = []
    # List of ages of patients, conditions and time stayed
    age_list = []
    condition_list = []
    time_stay_list = []
    fail_time = None


# Stores individual patient attributes
class Patient:
    def __init__(self):
        GlobalVars.patient_count += 1
        self.id = GlobalVars.patient_count
        self.time_in = GlobalVars.time
        self.time_stay = np.random.randint(10, 30) # will be DAI prediction
        self.time_out = 0
        self.status = ''
        self.age = np.random.randint(18, 70)
        self.condition = ''


async def show_charts(q: Q):
    del q.page['main'], q.page['menu'], q.page['notification']
    if len(GlobalVars.time_list) > 1:

        q.page['app_logo'] = ui.markup_card(
            box=app_config.plot01_box,
            title='',
            content="""<p style='text-align:center; vertical-align: top; display: table-cell; width: 134px;'>"""
                    """<a href='https://www.h2o.ai/h2o-q/'> <img src='""" + q.app.app_icon_url + """' height='40px' width='40px'> </a> </p>"""

        )

        # Sim status
        #q.page['plot01'] = ui.small_stat_card(box=app_config.plot02_box, title='Simulation Status',
        #                                       value='Simulating')

        pct = GlobalVars.time / q.app.simulation_time
        if pct > 1:
            pct = 1

        q.page['plot01'] = ui.wide_gauge_stat_card(box=app_config.plot02_box, title='Simulation Status',
                                                   value='Simulating', aux_value='', progress=pct, plot_color='$green')
        # Pat count stats
        q.page['plot02'] = ui.small_stat_card(box=app_config.plot03_box, title='Patient Count',
                                              value=f'{GlobalVars.patient_count-1}')
        # Pat wait count stats
        q.page['plot03'] = ui.small_stat_card(box=app_config.plot04_box, title='Wait Count',
                                              value=f'{GlobalVars.wait_count}')
        # Bed capacity stats
        q.page['plot04'] = ui.small_stat_card(box=app_config.plot05_box, title='Bed Capacity',
                                              value=f'{GlobalVars.bed_capacity}')
        # Forecast stats
        q.page['plot05'] = ui.small_stat_card(box=app_config.plot06_box, title='Forecasted Beds',
                                              value=f'{GlobalVars.forecasted_beds}')
        # Fail time stats
        q.page['plot06'] = ui.small_stat_card(box=app_config.plot07_box, title='Fail Time (days)',
                                              value=f'{GlobalVars.fail_time}')

        # Line chart for available beds
        rows = [(GlobalVars.time_list[i], GlobalVars.bed_avail_list[i]) for i in range(len(GlobalVars.time_list))]
        q.page['plot1'] = ui.plot_card(
                box=app_config.plot1_box,
                title='Hospital Bed Availability',
                data=data('time beds', rows=rows),
                plot=ui.plot([ui.mark(type='line', x='=time', y='=beds', x_min=0, y_min=0-GlobalVars.bed_capacity, y_max=GlobalVars.bed_capacity+50,
                                      x_title='Time', y_title='Hospital Bed Availability', color='#F39C12'),
                              ui.mark(y=GlobalVars.bed_capacity, label='Max Capacity', color=''),
                              ui.mark(y=0, label='Min Capacity', color='')]
                             ))

        # Histogram of age
        unique, counts = np.unique(GlobalVars.age_list, return_counts=True)
        rows = list(zip(unique.tolist(), counts.tolist()))
        q.page['plot21'] = ui.plot_card(
                box=app_config.plot21_box,
                title='Histogram of Age',
                data=data('age count', rows=rows),
                plot=ui.plot([ui.mark(type='interval', x='=age', y='=count', x_min=0, y_min=0,
                                      x_title='Age', y_title='Count', color='#F4D03F')]
                             ))

        # Histogram of conditions
        unique, counts = np.unique(GlobalVars.condition_list, return_counts=True)
        rows = list(zip(unique.tolist(), counts.tolist()))
        q.page['plot22'] = ui.plot_card(
                box=app_config.plot22_box,
                title='Histogram of Conditions',
                data=data('condition count', rows=rows),
                plot=ui.plot([ui.mark(type='interval', x='=condition', y='=count', x_min=0, y_min=0,
                                      x_title='condition', y_title='Count', color='#EC7063')]
                             ))


        # Stat chart
        # Plot baseline chart once and add to it for future iterations
        if not q.app.stats_plotted:
            plot_stat_card(q)
            q.app.stats_plotted = True

        q.app.c.data.qux = int(np.average(GlobalVars.time_stay_list))
        q.app.c.data.quux = (GlobalVars.time_stay_list[-1] - GlobalVars.time_stay_list[-2])/ 100
        q.app.c.plot_data[-1] = [int(np.average(GlobalVars.time_stay_list))]

        await q.page.save()
    else:
        q.page['plot1'] = ui.form_card(box=app_config.main_box, items=[ui.progress('Generating plots')])
        await q.page.save()


# Check if patient has been discharged and update patient attributes
def update_patients():
    # Track min bed count
    if GlobalVars.bed_count <= GlobalVars.min_beds:
        GlobalVars.min_beds = GlobalVars.bed_count

    # Only updated forecasted bed if it is less than bed capacity
    if GlobalVars.forecasted_beds <= GlobalVars.bed_capacity:
        GlobalVars.forecasted_beds = GlobalVars.bed_capacity - GlobalVars.min_beds

    # Check if discharged and update bed count
    for pat_id, patient in GlobalVars.all_patients.items():
        if patient.status == 'Admitted' and  GlobalVars.time > patient.time_out:
            patient.status = 'Discharged'
            if GlobalVars.bed_count <= GlobalVars.bed_capacity:
                GlobalVars.bed_count += 1
    # Recheck if patient was waiting and add to admission if true
    for pat_id, patient in GlobalVars.all_patients.items():
        if patient.status == 'Waiting' and GlobalVars.bed_count > 0:
            patient.status = 'Admitted'
            patient.time_out = GlobalVars.time + patient.time_stay
            GlobalVars.wait_count -= 1

def process_patients(row: None):
    # Update patient information
    p = Patient()
    # Check if beds available. If avail: patient is admitted and bed count is decreased.
    # If not, patient is in wait queue
    if GlobalVars.bed_count > 0:
        GlobalVars.bed_count -= 1
        p.status = 'Admitted'
        p.time_out = p.time_in + p.time_stay
    else:
        p.status = 'Waiting'
        GlobalVars.forecasted_beds += 1
        GlobalVars.wait_count += 1
        if not GlobalVars.fail_time:
            GlobalVars.fail_time = GlobalVars.time
    if not test_mode:
        p.age = row[patient_generator.age_col]
        p.condition = row[patient_generator.condition_col]
        p.time_stay = row['Predicted stay']

    # Common
    # Update global vars
    GlobalVars.time_list.append(GlobalVars.time)
    GlobalVars.bed_avail_list.append(GlobalVars.bed_count)
    GlobalVars.age_list.append(p.age)
    GlobalVars.time_stay_list.append(p.time_stay)
    GlobalVars.all_patients[p.id] = p
    GlobalVars.condition_list.append(p.condition)


async def trigger_admissions(q: Q):
    await show_charts(q)
    # Check for patient discharge
    update_patients()
    # Add patient to queue
    if not test_mode:
        pat_df = q.app.pat_df.iloc[q.app.start_row:q.app.end_row, :]
        q.app.start_row = q.app.end_row
        q.app.end_row = q.app.start_row + q.app.incoming_population_size
        fname = './score_patient.csv'
        pat_df.to_csv(fname, index=False)
        pat_df['Predicted stay'] = get_mojo_preds(fname)
        for i, row in pat_df.iterrows():
            process_patients(row)
    else:
        process_patients(None)


def show_patients():
    print('-----------')
    print(
        f'Available Beds={GlobalVars.bed_count}/{GlobalVars.bed_capacity}, patient_count = {GlobalVars.patient_count}')
    for pat_id, patient in GlobalVars.all_patients.items():
        print(f'id: {patient.id}, Discharged:{patient.discharged}')


# Plot a stat card
def plot_stat_card(q: Q):
    val = 0
    pc = 0
    # Avg time stay
    q.app.c = q.page.add('plot31', ui.wide_series_stat_card(
        box=app_config.plot08_box,
        title='Avg Stay (Days)',
        value='={{intl qux minimum_fraction_digits=2 maximum_fraction_digits=2}}',
        aux_value='={{intl quux style="percent" minimum_fraction_digits=1 maximum_fraction_digits=1}}',
        data=dict(qux=val, quux=pc),
        plot_type='area',
        plot_value='qux',
        plot_color='$red',
        plot_data=data('qux', -25),
        plot_zero_value=0,
        plot_curve='smooth'))


async def update_vars(q: Q):
    if GlobalVars.bed_count <= 0:
        print('WARNING: Bed count is below 0')
        q.app.limit_reached = True

    # Increase tick by random amount between user defined range
    interval = np.random.randint(q.app.arrival_interval[0], q.app.arrival_interval[1])
    GlobalVars.time += interval
    try:
        await trigger_admissions(q)
    except Exception as e:
        print(f'{e} ERROR')


# Setup license file in
async def setup_mojo(q: Q):
    # Write DAI license to scoring folder
    path = os.path.join(app_config.scoring_path, "license.sig")
    os.system(f'touch {path}')
    fout = open(path, "w")
    txt = q.app.dai_license
    fout.write(txt.strip())


async def setup_data(q: Q):
    # Show progress bar
    q.page['main'] = ui.form_card(box=app_config.main_box,
                                  items=[ui.progress(f'Generating synthetic patient data for a population size of {q.app.population_size}'),
                                         ui.text(app_config.synthea_img)
                                         ])
    await q.page.save()
    # Get synthetic patient data
    if not test_mode:
        q.app.pat_df = get_patients(q)
        pat_table = table_from_df(q.app.pat_df, 'pat_table', sortable=True, searchable=True, groupable=True)
        q.page['main'].items = [ui.text_xl('Patient Table'),
                                ui.button(name='start_sim', label='Simulate', primary=True),
                                pat_table
                                ]
    else:
        q.page['main'].items = [
                                ui.button(name='start_sim', label='Simulate', primary=True),
                                ]
    del q.page['menu']
    await q.page.save()


# Run simulation loop
async def run_loop(q: Q):
    # Initialize global app vars
    GlobalVars.patient_count = 0
    GlobalVars.wait_count = 0
    GlobalVars.time = 0
    GlobalVars.bed_capacity = q.app.bed_capacity
    GlobalVars.bed_count = q.app.bed_capacity
    GlobalVars.min_beds = q.app.bed_capacity
    GlobalVars.forecasted_beds = 0
    GlobalVars.fail_time = None
    GlobalVars.time_list = []
    GlobalVars.bed_avail_list = []
    GlobalVars.age_list = []
    GlobalVars.condition_list = []
    GlobalVars.time_stay_list = []
    GlobalVars.all_patients = {}

    q.app.start_row = 0
    q.app.end_row = q.app.incoming_population_size

    # Clear previous cards
    del q.page['menu'],  q.page['plot1']

    # Run loop till simulation time end
    if test_mode:
        while (GlobalVars.time <= q.app.simulation_time):
            await update_vars(q)
    else:
        while (GlobalVars.time <= q.app.simulation_time) and (q.app.start_row <= q.app.pat_df.shape[0]):
            await update_vars(q)

    # Show completion/success if sim completes and limit not reached
    if (GlobalVars.time > q.app.simulation_time):
        if q.app.limit_reached:
            status_message = 'Failure'
            q.page['plot01'].plot_color = '$red'
        else:
            status_message = 'Success'
        q.page['plot01'].value = status_message

        q.page['button_bar'] = ui.form_card(box=app_config.button_bar_box,
                                            items=[ui.buttons(
                                                [ui.button(name='restart_sim', label='Restart', primary=True),
                                                 ])])
        await q.page.save()
