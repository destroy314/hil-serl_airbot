from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.insert_vertical.config import TrainConfig as InsertVerticalTrainConfig
from experiments.timing_belt.config import TrainConfig as TimingBeltTrainConfig
from experiments.dual_airbot.config import TrainConfig as DualAirbotTrainConfig
from experiments.airbot_cart.config import TrainConfig as AirbotCartTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "insert_vertical": InsertVerticalTrainConfig, # single arm rel cart ctrl by spacemouse
                "timing_belt": TimingBeltTrainConfig, # abs joint ctrl
                "dual_airbot": DualAirbotTrainConfig, # rel joint ctrl
                "airbot_cart": AirbotCartTrainConfig, # rel cart ctrl by aribot expert
               }