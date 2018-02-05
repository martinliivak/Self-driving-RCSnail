import numpy as np
import cv2
from car_interfacing import CarConnection
from keras.models import load_model
from decimal import *
from time import time

# oka mudel: conv_dense_gear_bigdata

# conv_speshul_data_000001regL2_05dropout_LR
model = load_model(".\Models\conv_dense_gear_bigdata.h5")
connection = CarConnection(machine_name="tigu6")

frame_counter = 0
processed_frames = []

while True:
    try:
        # Kill condition counter
        frame_counter += 1

        # Single loopy time measurement
        start = time()

        # Connect to car and retrieve rescaled frame
        frame = connection.receive_data_from_stream()

        #cv2.imshow("img", frame)
        #cv2.waitKey(1)

        if frame is not None:
            # Normalizing frame
            normed = frame / 255
            processed_frames.append(normed)

            # Stack frames and process
            if len(processed_frames) >= 4:
                stacked_image = np.concatenate(processed_frames, axis=2)
                processed_frames.pop(0)

                prediction = model.predict(stacked_image[np.newaxis, :])
                pred_list = prediction.tolist()[0]

                print("original")
                for pred in pred_list:
                    print(str(round(Decimal(pred), 2)) + "\t", end="")
                print("")

                # Disallow too sudden turns (semi-bad idea)
                # pred_list[0] = -1 * pred_list[0]
                # if pred_list[0] >= -0.5 else (pred_list[0]/abs(pred_list[0])) * 0.5

                # Eliminate braking element if it's too small
                pred_list[1] = pred_list[1] if pred_list[1] >= 0.6 else 0.0

                # Normalize throttle to be always an effective input (0.3 or so minimum)
                # pred_list[2] = np.clip(pred_list[2], 0.0, 1.0)
                pred_list[2] = 0.2 + 0.6 * (1 - np.exp(-2.5 * np.clip(pred_list[2], 0.0, 1.0)))

                # Round gear to forward or backward gear
                pred_list[3] = 2 if pred_list[3] >= 1.3 else 1

                for pred in pred_list:
                    print(str(round(Decimal(pred), 2)) + "\t", end="")

                connection.send_commands_to_car(pred_list, steering_only=False)
                print("\n----------- time: " + str(time() - start))

        # Some breaking condition to kill me here
        if frame_counter >= 25000:
            break

    except KeyboardInterrupt:
        print('Connection killed.')
        connection.close()

print('Connection killed.')
connection.close()
