import datetime
import os
from sre_constants import BRANCH
import prefect
from pathlib import Path
from prefect import task
from prefect.engine.signals import SKIP
from prefect.tasks.shell import ShellTask

#prefect.context.config.home_dir = "/home/marcos/.prefect"


@task
def curl_cmd(url: str, fname: str) -> str:
    """
    The curl command we wish to execute.
    """
    if os.path.exists(fname):
        raise SKIP("Image data file already exists.")
    return "curl -fL -o {fname} {url}".format(fname=fname, url=url)


# ShellTask is a task from the Task library which will execute a given command in a subprocess
# and fail if the command returns a non-zero exit code

download = ShellTask(name="curl_task", max_retries=2, retry_delay=datetime.timedelta(seconds=10))

@task(skip_on_upstream_skip=False)
def load_and_split(fname: str) -> list:
    """
    Loads image data file at `fname` and splits it into
    multiple frames.  Returns a list of bytes, one element
    for each frame.
    """
    with open(fname, "rb") as f:
        images = f.read()

    return [img for img in images.split(b"\n" * 4) if img]


@task
def write_to_disk(image: bytes) -> bytes:
    """
    Given a single image represented as bytes, writes the image
    to the present working directory with a filename determined
    by `map_index`.  Returns the image bytes.
    """
    frame_no = prefect.context.get("map_index")
    with open("frame_{0:0=2d}.gif".format(frame_no), "wb") as f:
        f.write(image)
    return image

import imageio
from io import BytesIO


@task
def combine_to_gif(image_bytes: list) -> None:
    """
    Given a list of ordered images represented as bytes,
    combines them into a single GIF stored in the present working directory.
    """
    images = [imageio.imread(BytesIO(image)) for image in image_bytes]
    imageio.mimsave('./clip.gif', images)

from prefect import Parameter, Flow


DATA_URL = Parameter("DATA_URL",
                     default="https://github.com/cicdw/image-data/blob/master/all-images.img?raw=true")

DATA_FILE = Parameter("DATA_FILE", default="image-data.img")


with Flow("Image ETL Marcos") as flow:

    # Extract
    command = curl_cmd(DATA_URL, DATA_FILE)
    curl = download(command=command)

    # Transform
    # we use the `upstream_tasks` keyword to specify non-data dependencies
    images = load_and_split(fname=DATA_FILE, upstream_tasks=[curl])

    # Load
    frames = write_to_disk.map(images)
    result = combine_to_gif(frames)


# start our Dask cluster
#from dask.distributed import Client


#client = Client(n_workers=1, threads_per_worker=1)

# point Prefect's DaskExecutor to our Dask cluster

from prefect.executors import DaskExecutor
if __name__ == '__main__':
    from prefect import task, Flow
    from prefect.storage import GitLab

    flow.storage = GitLab(
        repo="FACE_IFCA",                            # name of repo
        path="face_ifca/Prefect/Clase/local_agent/5-daskexecutor.py",                    # location of flow file in repo
        #access_token_secret="gitlab+deploy-token-1148584"   # name of personal access token secret
    )
    executor = DaskExecutor('tcp://193.146.75.210:64846')
    #flow.run(executor=executor)
    flow.visualize()
    #flow.executor=executor
    flow.register('Aida', add_default_labels=False, labels=["mods-prefect"])
