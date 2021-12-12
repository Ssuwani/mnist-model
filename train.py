import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.python.lib.io import file_io
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_units", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--save_model_bucket_name", type=str, default=False)

    args = parser.parse_args()
    return args


def save_model(model, hidden_units, optimizer, acc, bucket_name):
    bucket_path = os.path.join(bucket_name, "mnist")
    save_path = f"units_{hidden_units}_opt_{optimizer}.h5"
    print(f"saving model {save_path}")
    model.save(save_path)

    gs_path = os.path.join(bucket_path, save_path)
    with file_io.FileIO(save_path, mode="rb") as input_file:
        with file_io.FileIO(gs_path, mode="wb+") as output_file:
            output_file.write(input_file.read())
    print(f"model save success!")

    # request_deploy_api(gs_path)
    # print(f"Trigger Deploy success!")


def train(hidden_units, optimizer, bucket_name):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    inputs = Input(shape=(28, 28))
    x = layers.Flatten()(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.get(optimizer)

    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    model.fit(train_x, train_y, epochs=3, validation_split=0.2)
    loss, acc = model.evaluate(test_x, test_y)
    print(f"model test-loss={loss:.4f} test-acc={acc:.4f}")

    if bucket_name:
        save_model(model, hidden_units, optimizer, acc, bucket_name)

    if save_model:
        tf.saved_model.save(
            model, f"./saved_model/mnist-{hidden_units}-{optimizer}-{acc*100}"
        )


if __name__ == "__main__":
    args = get_args()
    train(args.hidden_units, args.optimizer, args.save_model_bucket_name)
