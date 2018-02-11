import cv2
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from keras.utils import Sequence
import math
import threading


def cut_labels_for_video(video_file, label_file, cut_from_front):
    """
    If cut_from_front then remove a chunk of labels from the front, otherwise remove it from the back."""

    training_video = cv2.VideoCapture(video_file)
    if "_clean" in label_file:
        training_labels = pd.read_csv(label_file, sep="\t")
    else:
        training_labels = pd.read_csv(label_file, sep="\t", decimal=",")

    # Unfortunately cap.get(cv2.CAP_PROP_FRAME_COUNT) gave bullshit results
    video_length = count_frames_manually(training_video)
    label_length = training_labels.shape[0]
    diff = label_length - video_length

    print("Video len: " + str(video_length))
    print("Label len: " + str(label_length))

    if diff <= 0:
        print("Video is longer than labels. Check your filenames or something.")
        return
    print("Dropping " + str(diff) + " lines from labels.")

    if cut_from_front:
        training_labels.drop(training_labels.index[0:diff], inplace=True)
    else:
        training_labels.drop(training_labels.index[-diff:], inplace=True)

    assert video_length == training_labels.shape[0]

    label_filename = label_file.split("/")[-1].split(".")[0]

    if "_clean" in label_filename:
        training_labels.to_csv("./Data/Preprocessed/" + label_filename + ".csv", sep="\t", index=False)
    else:
        training_labels.to_csv("./Data/Preprocessed/" + label_filename + "_clean.csv", sep="\t", index=False)


def count_frames_manually(video):
    """
    It's a bit slooow, eeeh?"""
    total = 0

    while True:
        result, frame = video.read()

        if not result:
            break

        total += 1
    return total


def extract_training_data(filename, csv_filename, image_size=(64, 64, 3)):
    """
    Read every frame from input video and output them.

    :param filename:
    :param csv_filename:
    :param image_size:
    :return: images as flattened lists and training labels
    """
    cap = cv2.VideoCapture(filename)
    labels = pd.read_csv(csv_filename, sep="\t")

    frame_counter = 0
    processed_frames = []

    training_images = []
    training_label_ids = []

    while True:
        frame_counter += 1
        result, frame = cap.read()
        if result and frame_counter % 4 == 0:
            frame = frame / 255
            # cv2.imshow("img", frame)
            resized = cv2.resize(frame, image_size[:2])

            training_images.append(resized)
            training_label_ids.append(frame_counter)

        if cv2.waitKey(1) & 0xFF == ord('q') or not result:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    training_images = np.array(training_images)
    training_labels = labels.loc[training_label_ids]

    assert training_images.shape[0] == training_labels.shape[0]
    return training_images, training_labels


def extract_training_data_in_overlapping_groups(filename, csv_filename, image_size=(64, 64, 3)):
    """
    Read every 12th frame from input video and bundle every five frames together.

    :param filename:
    :param csv_filename:
    :param image_size:
    :return: images as flattened lists and training labels
    """
    cap = cv2.VideoCapture(filename)
    labels = pd.read_csv(csv_filename, sep="\t")

    frame_counter = 0
    processed_frames = []
    processed_frames_labels = []

    training_images = []
    training_labels = []
    while True:
        frame_counter += 1
        result, frame = cap.read()
        if result and frame_counter % 12 == 0:
            frame = frame / 255
            # cv2.imshow("img", frame)

            resized = cv2.resize(frame, image_size[:2])
            processed_frames.append(resized)
            processed_frames_labels.append(labels.loc[frame_counter].values)

            if len(processed_frames) >= 4:
                training_images.append(processed_frames.copy())
                training_labels.append(processed_frames_labels.copy())

                processed_frames.pop(0)
                processed_frames_labels.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q') or not result:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)

    assert training_images.shape[0] == training_labels.shape[0]
    return training_images, training_labels


def generate_multifile(video_files, label_files, image_size=(64, 64, 3), batch_size=64):
    assert len(video_files) == len(label_files), 'Length of video file list is not the same as label file list'

    while 1:
        batch_counter = 0
        batch_shape = (batch_size,) + image_size

        output_images = np.zeros(batch_shape)
        output_labels = np.zeros((batch_size, 3))

        for i in range(len(video_files)):
            frame_counter = 0

            labels = pd.read_csv('Data/Preprocessed/' + label_files[i], sep="\t")
            cap = cv2.VideoCapture('Data/Preprocessed/' + video_files[i])

            while True:
                result, frame = cap.read()
                if result and frame_counter % 12 == 0:
                    frame = frame / 255.

                    resized_frame = cv2.resize(frame, image_size[:2])

                    output_images[batch_counter] = resized_frame.copy()
                    output_labels[batch_counter] = labels.loc[frame_counter].values[2:5]

                    batch_counter += 1

                    if batch_counter == batch_size:
                        yield (output_images, output_labels)

                        batch_counter = 0
                        output_images = np.zeros(batch_shape)
                        output_labels = np.zeros((batch_size, 3))

                if cv2.waitKey(1) & 0xFF == ord('q') or not result:
                    break

                frame_counter += 1

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()


def run_kfold_cross_val(build_fn, x_train, y_train, epochs=10, batch_size=64, verbose=0, n_splits=10):
    model = KerasRegressor(build_fn=build_fn, epochs=epochs, batch_size=batch_size, verbose=verbose)
    kfold = KFold(n_splits=n_splits)

    return cross_val_score(model, x_train, y_train, cv=kfold, scoring='explained_variance')


class SnailSequence(Sequence):
    def __init__(self, video_file, labels_file, batch_size=64, image_size=(64, 64, 3), every_nth=12):
        self.batch_size = batch_size
        self.image_size = image_size
        self.video_file = video_file
        df = pd.read_csv('Data/' + labels_file, sep='\t')
        self.y = df.iloc[::every_nth, :]
        self.x = self.y.FrameNo.values

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        batch_x_frames = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        # Steering braking throttle -> 2:5, to include gear, 2:6
        batch_y = self.y.iloc[idx * self.batch_size: (idx + 1) * self.batch_size, 2:5].values

        batch_x = np.zeros((batch_y.shape[0],) + self.image_size)

        # Get the frames from the video
        for i, frame_no in enumerate(batch_x_frames):
            cap = cv2.VideoCapture('Data/' + self.video_file)
            cap.set(1, frame_no)
            result, frame = cap.read()

            frame = frame / 255.
            resized_frame = cv2.resize(frame, self.image_size[:2])
            batch_x[i] = resized_frame.copy()

            cap.release()
            cv2.destroyAllWindows()

        return batch_x, batch_y


def get_partial_batch_stacked(video, label_file, image_size):
    cap = cv2.VideoCapture('Data/Preprocessed/' + video)
    labels = pd.read_csv('Data/Preprocessed/' + label_file, sep='\t')

    frame_counter = 0
    processed_frames = []

    while True:
        result, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q') or not result:
            break

        if result and frame_counter % 12 == 0:
            resized_frame = cv2.resize(frame, image_size[:2])
            resized_frame = resized_frame / 255.

            processed_frames.append(resized_frame)

            if len(processed_frames) >= 4:
                stacked_image = np.concatenate(processed_frames, axis=2)
                yield stacked_image, labels.iloc[frame_counter].values[1:5]

                processed_frames.pop(0)

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


def generate_multifile_conc(video_files, label_files, image_size=(64, 64, 3), batch_size=64, nr_batches=10000):
    assert len(video_files) == len(label_files), 'Length of video file list is not the same as label file list'

    entries_per_cap = batch_size // len(video_files)

    while 1:
        for _ in range(nr_batches):
            batch_x = []
            batch_y = []

            generators = []
            for i in range(len(video_files)):
                generators.append(get_partial_batch_stacked(video_files[i], label_files[i], image_size))

            for i in range(len(generators)):
                for _ in range(entries_per_cap):
                    try:
                        x, y = next(generators[i])
                    except StopIteration:
                        generators[i] = get_partial_batch_stacked(video_files[i], label_files[i], image_size)
                        x, y = next(generators[i])

                    batch_x.append(x)
                    batch_y.append(y)

            yield np.array(batch_x), np.array(batch_y)

            batch_x = []
            batch_y = []


def random_file_gen(video_files, label_files, image_size=(64, 64, 3), batch_size=64):
    assert len(video_files) == len(label_files), 'Length of video file list is not the same as label file list'
    while 1:
        file_nr = np.random.randint(len(video_files) + 1)
        batch_x = []
        batch_y = []

        while True:
            while len(batch_x) < batch_size:
                try:
                    x, y = next(get_partial_batch_stacked(video_files[file_nr], label_files[file_nr], image_size))
                    batch_x.append(x)
                    batch_y.append(y)
                except StopIteration:
                    pass

            yield batch_x, batch_y
            batch_x = []
            batch_y = []

