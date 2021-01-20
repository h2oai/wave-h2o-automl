from h2o_wave import ui, app, Q, main, expando_to_dict
import time
from typing import Collection
from typing import Optional
import mlops
import requests
from .utils import *
import json

# Shows table of DAI experiments
def show_dai_experiments(q: Q, card_name, card_box):
    driverless = q.user.dai_client
    experiments = driverless.experiments.list()
    experiment_names = []
    experiment_keys = []
    if experiments:
        for experiment in experiments:
            experiment_names.append(experiment.name)
            experiment_keys.append(experiment.key)
        experiments_df = pd.DataFrame({'Experiment Key': experiment_keys, 'Experiment Name': experiment_names})
    else:
        experiments_df = pd.DataFrame()
    data_table = table_from_df(experiments_df, 'dai_experiments_table', filterable=True, searchable=True, groupable=True)
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('DAI Experiments'),
        data_table
    ])
    return experiments_df


# Show list of projects in MLOps
def mlops_list_projects(q: Q, mlops_client:mlops.Client, card_name, card_box):
    projects = mlops_client.storage.project.list_projects(mlops.StorageListProjectsRequest()).project
    created_time = []
    desc = []
    names = []
    ids = []
    for project in projects:
        ids.append(project.id)
        names.append(project.display_name)
        desc.append(project.description)
        created_time.append(project.created_time)

    projects_df = pd.DataFrame({'id': ids, 'display_name': names, 'description': desc, 'created_time': created_time})
    mlops_proj_table = table_from_df(projects_df, 'mlops_projects_table', filterable=True, searchable=True, groupable=True)
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('MLOPs Projects'),
        mlops_proj_table
    ])
    return projects_df


# Show list of deployments for a given project in MLOps
def mlops_list_deployments(q: Q, mlops_client: mlops.Client, project_id, card_name, card_box):
    deployments = mlops_client.storage.deployment.list_deployments(mlops.StorageListDeploymentsRequest(project_id=project_id)).deployment
    created_time = []
    experiments = []
    projects = []
    ids = []
    for deployment in deployments:
        ids.append(deployment.id)
        projects.append(deployment.project_id)
        experiments.append(deployment.experiment_id)
        created_time.append(deployment.created_time)
    deployments_df = pd.DataFrame({'id': ids, 'experiment_id': experiments, 'project_id': projects, 'created_time': created_time})
    mlops_deployment_table = table_from_df(deployments_df, 'mlops_deployments_table', filterable=True, searchable=True, groupable=True)
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('Deployments'),
        mlops_deployment_table
    ])
    return deployments_df


# Deploy an experiment in MLOps
def mlops_deploy_experiment(
        mlops_client: mlops.Client,
        project_id: str,
        experiment_id: str,
        type: str,
        secondary_scorers: Optional[Collection[mlops.StorageSecondaryScorer]] = None,
        metadata=None,
        environment_name="DEV",
):
    """Deploys the experiment using the mlops_client."""
    svc = mlops_client.storage.deployment_environment
    envs = svc.list_deployment_environments(
        mlops.StorageListDeploymentEnvironmentsRequest(
            project_id=project_id,
        )
    ).deployment_environment
    dev_env = next((env for env in envs if env.display_name == environment_name), None)
    if dev_env is None:
        raise RuntimeError(f"Could not find '{environment_name}' environment.")

    return (
        svc.deploy(
            mlops.StorageDeployRequest(
                experiment_id=experiment_id,
                deployment_environment_id=dev_env.id,
                type=type,
                secondary_scorer=secondary_scorers,
                metadata=metadata,
            )
        )
    ).deployment


# Wait for deployment to be a specific status
def mlops_wait_for_status(
        statuses: Collection[int],
        req: requests.Request,
        max_retries: int = 10,
        sleep: int = 2,
):
    """Sends the request and retries until the response status code is one of statuses.
    Then it returns the response. Retries max_retries at maximum and waits sleep seconds
    between retries.
    Intended usage is with sample requests and scoring requests when deployment may
    look HEALTHY but is not yet accessible via ingress.
    """
    r: requests.Response
    prep = req.prepare()

    with requests.Session() as session:
        r = session.send(prep)
        for _ in range(max_retries):
            if r.status_code in statuses:
                break
            time.sleep(sleep)
            r = session.send(prep)

    return r


# Wait for deployment to be healthy
def mlops_deployment_should_become_healthy(deployment_id, mlops_client):
    svc = mlops_client.deployer.deployment_status
    status: mlops.DeployDeploymentStatus
    dead_line = time.monotonic() + 120

    i = 0
    while True:
        i = i + 1
        print("Deployment_should_become_healthy - ", str(i), flush=True)
        time.sleep(1)
        status = svc.get_deployment_status(
            mlops.DeployGetDeploymentStatusRequest(deployment_id=deployment_id)
        ).deployment_status
        if (
                status.state == mlops.DeployDeploymentState.HEALTHY
                or time.monotonic() > dead_line
        ):
            break

    assert status.state == mlops.DeployDeploymentState.HEALTHY
    return status


# Links DAI experiment to a DAI project
async def link_experiment_to_project(q: Q):
    driverless = q.user.dai_client
    # Create project
    q.user.project_key = driverless._backend.create_project(q.args.project_name, q.args.project_desc)
    # Link project to experiment
    driverless._backend.link_experiment_to_project(
        project_key=q.user.project_key,
        experiment_key=q.user.experiment_key,
    )
 

# Use MLOps to setup scorer
async def mlops_scorer_setup(q: Q):
    mlops_client = q.user.mlops_client
    deployment_id = q.user.deployment_id
    status = mlops_deployment_should_become_healthy(mlops_client=mlops_client,
                                              deployment_id=deployment_id)
    q.user.sample_url = status.scorer.sample_request.url
    q.user.score_url = status.scorer.score.url

    print(q.user.sample_url, q.user.score_url)


# Get a sample request for a deployment
def mlops_get_sample_request(sample_url):
    r = mlops_wait_for_status(
        [200],
        req=requests.Request(
            "GET",
            sample_url,
        ),
    )
    assert r.ok
    return json.dumps(r.json())


# Score using endpoint URL of a deployment
def mlops_get_score(score_url, query):
    query_json = json.loads(query)
    response = requests.post(url=score_url, json=query_json)
    assert response.status_code == 200
    return json.loads(response.text)


# Delete an MLOps project
def mlops_delete_project(q: Q, project_key):
    driverless = q.user.dai_client
    driverless._backend.delete_project(project_key)


# Undeploy, unlink experiment from project and delete project
def mlops_cleanup(q: Q):
    mlops_client = q.user.mlops_client
    driverless = q.user.dai_client
    project_key = q.user.project_key
    experiment_key = q.user.experiment_key

    mlops_client.storage.deployment.undeploy(
        mlops.StorageUndeployRequest(deployment_id=q.user.deployment_key)
    )

    driverless._backend.unlink_experiment_from_project(
        project_key=project_key,
        experiment_key=experiment_key
    )
    driverless._backend.delete_project(project_key)


