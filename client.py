import math
import argparse
from typing import Dict, List, Tuple
import flwr as fl

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)

NUM_CLIENTS = 50

def prepare_dataset():
    train_dir = "train" #passing the path with training images
    test_dir = "test"   #passing the path with testing images
    img_size = 48 #original size of the image

    train_datagen = ImageDataGenerator(#rotation_range = 180,
                                            width_shift_range = 0.1,
                                            height_shift_range = 0.1,
                                            horizontal_flip = True,
                                            rescale = 1./255,
                                            #zoom_range = 0.2,
                                            validation_split = 0.2
                                            )
    validation_datagen = ImageDataGenerator(rescale = 1./255,
                                            validation_split = 0.2)

    train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                        target_size = (img_size,img_size),
                                                        batch_size = 64,
                                                        color_mode = "grayscale",
                                                        class_mode = "categorical",
                                                        subset = "training"
                                                    )
    validation_generator = validation_datagen.flow_from_directory( directory = test_dir,
                                                                target_size = (img_size,img_size),
                                                                batch_size = 64,
                                                                color_mode = "grayscale",
                                                                class_mode = "categorical",
                                                                subset = "validation"
                                                                )
    
    partitions = []
    # We keep all partitions equal-sized in this example
    partition_size = math.floor(len(train_generator) / NUM_CLIENTS)
    for cid in range(NUM_CLIENTS):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

        train_generator_cid = train_generator[idx_from:idx_to]

        # now partition into train/validation
        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(train_generator_cid) * 0.9)

        client_train = train_generator_cid[:split_idx]
        client_val = train_generator_cid[split_idx:]
        partitions.append((client_train, client_val))

    return partitions, validation_generator

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainset, valset, use_mnist: bool):
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset

        self.model = model = keras.Sequential(
            [
                keras.Input(shape=(48, 48, 1)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.25),

                keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.25),

                keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.25),

                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.25),

                keras.layers.Dense(512, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.25),

                keras.layers.Dense(7, activation="softmax"),
            ]
        )

        self.model.compile(
                        optimizer = Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

    def get_parameters(self, config) -> List[np.ndarray]:
        return self.model.get_weights()
    
    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int]:
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Set hyperparameters from config sent by server/strategy
        batch, epochs = config["batch_size"], config["epochs"]
        # train
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch)
        return self.get_parameters({}), len(self.x_train), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[int, float, float]:
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Download CIFAR-10 dataset and partition it
    partitions, _ = prepare_dataset()
    trainset, valset = partitions[args.cid]

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(trainset=trainset, valset=valset),
    )


if __name__ == "__main__":
    main()