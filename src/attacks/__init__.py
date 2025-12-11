"""
Adversarial attack implementations for DINOmite.
"""

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.bim import bim_attack
from src.attacks.cw import carlini_wagner_attack
from src.attacks.deepfool import deepfool_attack_simple as deepfool_attack
from src.attacks.autoattack import autoattack_evaluate

__all__ = [
    "fgsm_attack",
    "pgd_attack",
    "bim_attack",
    "carlini_wagner_attack",
    "deepfool_attack",
    "autoattack_evaluate",
]
