# Install jtop to check the jetpack version and l4t version in https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
FROM  dustynv/ros:humble-ros-base-l4t-r35.3.1
# To make sure installing tzdata will not prompt questions
ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh neovim tmux git htop curl wget \
    net-tools iputils-ping\
    # init certificate for curl
    ca-certificates \
    # For x11 forwarding tests (xeyes, xclock etc.)
    x11-apps \
    build-essential libboost-all-dev libeigen3-dev \
    # For pytorch
    libopenblas-dev \
    cmake sudo

ARG USERNAME=real
ARG USER_UID=1000
ARG USER_GID=1000


# Add the new user (userid should be consistent to the one in the host machine)
# Change the password to "real". You can use other password by changing the following line
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd ${USERNAME}  -u ${USER_UID} -g ${USER_GID} -m -p "$(openssl passwd -1 real)"
# Add sudo access (not recommended, but is hulepful for development)
RUN usermod -aG sudo ${USERNAME}
# Give ownership of the user's home directory to the new user
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}
USER ${USERNAME}
# Set the user's home directory as the working directory
WORKDIR /home/${USERNAME}


############### Development Tools ###############

# Install pytorch. 
# It does not have access to cuda kernels as the Jetpack version on Unitree jetson is usually lower than 6.0.
# In our project, the whole-body controller only uses CPU and diffusion policy is run on a separate PC
#   thus cuda compute is not required in our project
RUN wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl && \
    pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl && \
    rm torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# To get torch working on earlier Jetpack version, you can try python3.8 versions in https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#   Only Python 3.8 is supported on versions earlier than Jetpack 6.0 

RUN pip install numpy && pip install typing-extensions --upgrade

