import logging
from datetime import datetime
from sys import stdout

__logger = logging.getLogger("data690.log")
__logger.setLevel(logging.DEBUG)

fForm = logging.Formatter("%(asctime)s -> [%(level)s] %(message)s")
cForm = logging.Formatter("[%(level)s] %(message)s")


fHand = logging.FileHandler(f"log/{datetime.now().strftime("%Y%m%d%H%M%S")}.log")
fHand.setLevel(logging.DEBUG)
fHand.setFormatter(fForm)
__logger.addHandler(fHand)

cHand = logging.StreamHandler(stdout)
cHand.setLevel(logging.DEBUG)
cHand.setFormatter(cForm)
__logger.addHandler(cHand)