
docker build -f Dockerfile -t deploy .

docker run \
  --network host \
  --volume=/var/run/spnav.sock:/var/run/spnav.sock \
  --runtime=nvidia \
  --volume="$(pwd):/home/real/deploy" \
  --name deploy \
  -it deploy

# Install unitree_sdk2_python
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip install -e .

