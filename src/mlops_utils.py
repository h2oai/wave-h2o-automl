from h2o_wave import ui, app, Q, main, expando_to_dict
import time
from typing import Collection
from typing import Optional
import mlops
import requests
from .utils import *


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
        q.user.experiments_df = pd.DataFrame({'Experiment Key': experiment_keys, 'Experiment Name': experiment_names})
    else:
        q.user.experiments_df = pd.DataFrame()
    data_table = table_from_df(q.user.experiments_df, 'dai_experiments_table', filterable=True, searchable=True, groupable=True)
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('DAI Experiments'),
        data_table
    ])


# Return list of projects
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

    q.user.projects_df = pd.DataFrame({'id': ids, 'display_name': names, 'description': desc, 'created_time': created_time})
    mlops_proj_table = table_from_df(q.user.projects_df, 'mlops_projects_table')
    q.page[card_name] = ui.form_card(box=card_box, items=[
        ui.text_xl('MLOPs Projects'),
        mlops_proj_table
    ])


def mlops_list_deployments(q: Q, mlops_client : mlops.Client, project_id):
    status = mlops_client.storage.deployment.list_deployments(mlops.StorageListDeploymentsRequest(project_id)).deployment
    """
    status = mlops.DeployListDeploymentStatusesResponse = (
        mlops_client.deployer.deployment_status.list_deployment_statuses(
            mlops.DeployListDeploymentStatusesRequest(project_id=project_id)
        )).deployment_status
    """
    print(f'TKTK!! {status}')


# Functions below leveraged from tomk's example
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
    look HELTHY but is not yet accessible via ingress.
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


def mlops_deployment_should_become_healthy(deployment_id, mlops_client):
    svc = mlops_client.deployer.deployment_status
    status: mlops.DeployDeploymentStatus
    dead_line = time.monotonic() + 120

    i = 0
    while True:
        i = i + 1
        print("deployment_should_become_healthy - ", str(i), flush=True)
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
    experiment_index = int(q.args.dai_experiments_table[0])
    q.user.experiment_key = q.user.experiments_df['Experiment Key'][experiment_index]
    # Create project
    q.user.project_key = driverless._backend.create_project("test-project", "Test Project")
    # Link project to experiment
    driverless._backend.link_experiment_to_project(
        project_key=q.user.project_key,
        experiment_key=q.user.experiment_key,
    )


# Deploy an experiment to MLOps
async def mlops_deploy(q: Q):
    mlops_client = q.user.mlops_client
    project_key = q.user.project_key
    experiment_key = q.user.experiment_key

    deployment = mlops_deploy_experiment(
        mlops_client=mlops_client,
        project_id=project_key,
        experiment_id=experiment_key,
        type=mlops.StorageDeploymentType.SINGLE_MODEL
    )
    q.user.deployment_id = deployment.id


# Use MLOps to setup scorer
async def mlops_scorer_setup(q: Q):
    mlops_client = q.user.mlops_client
    deployment_id = q.user.deployment_id
    status = mlops_deployment_should_become_healthy(mlops_client=mlops_client,
                                              deployment_id=deployment_id)
    q.user.sample_url = status.scorer.sample_request.url
    q.user.score_url = status.scorer.score.url

    print(q.user.sample_url, q.user.score_url)

## MLOps predict
#async def mlops_predict(q: Q):
    sample_url = q.user.sample_url
    score_url = q.user.score.url

    r = mlops_wait_for_status(
        [200],
        req=requests.Request(
            "GET",
            sample_url,
        ),
    )
    assert r.ok
    print(r.json(), flush=True)
    response = requests.post(url=score_url, json=r.json())
    assert response.status_code == 200
    print(response.text, flush=True)

    return response.text


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


