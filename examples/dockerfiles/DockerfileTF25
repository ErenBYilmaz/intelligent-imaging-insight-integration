FROM tensorflow/tensorflow:2.5.0-gpu

# run with:
# docker build -t ${USER}_container_tf2 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -f DockerfileTF25
# docker run -it --rm --gpus all --user "$(id -u):$(id -g)" --ipc=host -v $(pwd):/code ${USER}_container_tf2

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# otherwise some nividia stuff will not install properly
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list  # not needed for tf >= 2.11

# some useful tools
RUN apt-get update && apt-get install curl git unzip graphviz nano iputils-ping -y && rm -rf /var/lib/apt/lists/*


# create current user inside image, so that newly created files belong to us and not to root and can be accessed outside docker as well
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && usermod -a -G root user && addgroup --gid 1011 data-acc && usermod -a -G data-acc user
USER user
WORKDIR /home/user

# Create environment
RUN pip install --upgrade pip
RUN pip install --upgrade cmake
RUN pip install hanging_threads colorama matplotlib numpy pandas cachetools tabulate yappi graphviz joblib scipy gitpython markdown2 seaborn psutil pydot numba scikit-learn scikit-image pygments natsort voluptuous humanfriendly coloredlogs simpleitk itk openpyxl jupyterlab jupyter "xlrd < 2" pydot pydicom click sympy lifelines dill plotly "wandb < 0.15.12" pycryptodome schedule astropy astroquery tqdm
RUN pip install tensorflow-addons==0.13.0 tensorflow_probability==0.13.0 tensorboard_plugin_profile
RUN pip install --no-dependencies astroNN  # it specifies a newer tensorflow version but works with 2.5.0


WORKDIR /code