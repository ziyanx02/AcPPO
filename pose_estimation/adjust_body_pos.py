import os
import yaml

from utils.interaction import Interact

cfg = yaml.safe_load(open("./cfgs/go2.yaml"))
interact = Interact(cfg)
interact.run()