"""Vendored, pure-tensor AB-MIL model cores (faithful to the validated study)."""

from cascadir.models.abmil import AbMil
from cascadir.models.attention import AttentionModule
from cascadir.models.bag_classifier import BagClassifier
from cascadir.models.instance_encoder import InstanceEncoder

__all__ = ["AbMil", "AttentionModule", "BagClassifier", "InstanceEncoder"]
