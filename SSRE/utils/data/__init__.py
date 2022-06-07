from utils.data.prepare_data import prepare_CL
from utils.data.prepare_loader import prepare_loader_normal


datatypes = {
    'cl': prepare_CL,
}

loadertypes = {
    'normal': prepare_loader_normal,
}

