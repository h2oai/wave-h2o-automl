from h2o_wave import main, app, Q, ui, data
import os
from .synthea_config import *


# Return dataframe for all patients given user defined settings
def get_patients(q: Q):
    cwd = os.getcwd()
    synthea_jar = cwd + '/src/simulators/synthea-with-dependencies.jar'
    output_folder = cwd + '/src/simulators/synthea_output'
    population_size = q.app.population_size
    age_range = str(q.app.age_range[0]) + '-' + str(q.app.age_range[1])
    # Execute synthea simulator with options
    modules = ':'.join(q.app.conditions)
    cmd = f"java -jar {synthea_jar} -a {age_range} -p {population_size} -m {modules} --exporter.csv.export true --exporter.baseDirectory {output_folder}"
    patient_generator.output_folder = output_folder + '/csv/'

    os.system(cmd)
    # Get patient file after execution
    patient_df = patient_generator.get_patient_df()
    return patient_df

