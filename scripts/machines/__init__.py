
from .BasicMachine import BasicMachine
from .VX import VX
from .S2AM import S2AM

def basic(**kwargs):
	return BasicMachine(**kwargs)

def s2am(**kwargs):
    return S2AM(**kwargs)

def vx(**kwargs):
    return VX(**kwargs)
