import simpy
import numpy as np
import os
from synthea_config import *
import pandas as pd
from h2o_wave import main, app, Q, ui
env = simpy.Environment()

def get_patient():
    cwd = os.getcwd()
    synthea_jar = cwd + '/synthea-with-dependencies.jar'
    output_folder = cwd + '/synthea_output'
    population_sze = 1
    age_range = '18-80'
    # Execute synthea simulator with options
    cmd = f"java -jar {synthea_jar} -a {age_range} -p {population_sze} --exporter.csv.export true --exporter.baseDirectory {output_folder}"
    patient_generator.output_folder = output_folder + '/csv/'

    pat = os.system(cmd)
    # Get patient file after execution
    patient_df = patient_generator.get_patient_df()
    return patient_df


# Global variables
class GlobalVars:
    patient_count = 0
    all_patients = {}
    bed_capacity = 100
    bed_count = bed_capacity


# Stores individual patient attributes
class Patient:
    def __init__(self, env):
        GlobalVars.patient_count += 1
        GlobalVars.bed_count -= 1
        self.id = GlobalVars.patient_count
        self.time_in = env.now
        self.time_stay = np.random.randint(4,5)
        self.time_out = self.time_in + self.time_stay
        self.discharged = False
        self.age = np.random.randint(18,70)


# Check if patient has been discharged and update patient attributes
def process_patients(env):
    for pat_id, patient in GlobalVars.all_patients.items():
        if env.now > patient.time_out:
            patient.discharged = True
            if GlobalVars.bed_count < GlobalVars.bed_capacity:
                GlobalVars.bed_count += 1




def trigger_admissions(env):
    while True:
        if GlobalVars.bed_count < 0:
            print('WARNING: Bed count is below 0')
            break

        pat = get_patient()

        # Check for patient discharge
        process_patients(env)
        # Add patient to queue
        p = Patient(env)

        print(f'Patient arriving at {env.now} Bed count={GlobalVars.bed_count}, patient_count = {GlobalVars.patient_count}')

        # Add to dict
        GlobalVars.all_patients[p.id] = p
        patient_inflow_time = np.random.randint(1, 4)
        yield env.timeout(patient_inflow_time)


def show_patients():
    print('-----------')
    print(f'Available Beds={GlobalVars.bed_count}/{GlobalVars.bed_capacity}, patient_count = {GlobalVars.patient_count}')
    for pat_id, patient in GlobalVars.all_patients.items():
        print(f'id: {patient.id}, Discharged:{patient.discharged}')


def run_loop():
    env.process(trigger_admissions(env))
    env.run(until=15)
    return



#run_loop()
time_list = [0, 4 , 5, 6]
bed_avail_list = [100, 130, 23, 45]
data = [(time_list[i], bed_avail_list[i]) for i in range(len(time_list))]
print('tk')