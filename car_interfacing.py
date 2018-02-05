import json
import socket
from PIL import Image, ImageFile
from io import BytesIO
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarConnection(object):
    def __init__(self, machine_name='tigu6'):
        remote_ip = socket.gethostbyname(machine_name)
        send_server = (remote_ip, 22241)
        receive_server = (remote_ip, 22242)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect(send_server)

        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_sock.connect(receive_server)

    def send_commands_to_car(self, commands, steering_only=False):
        """
        Method to send the movement commands to the relevant RCSnail car
        :param commands: a list of commands, in order steering (double), braking (double), throttle (double), gear (int)
        :return:
        """
        if steering_only:
            data = {'steering': commands[0]}

        else:
            assert len(commands) == 4, 'The input array must have all 4 parameters'

            data = {
                'steering': commands[0],
                'braking': commands[1],
                'throttle': commands[2],
                'gear': commands[3]
            }

        json_body = json.dumps(data)
        self.send_sock.sendall((json_body + '\n').encode('utf-8'))

    def receive_data_from_stream(self):
        header = self.recv_sock.recv(4)
        bytes_to_read = int.from_bytes(header, byteorder='little')

        buf = bytearray(bytes_to_read)
        view = memoryview(buf)

        while bytes_to_read:
            nbytes = self.recv_sock.recv_into(view, bytes_to_read)
            view = view[nbytes:]
            bytes_to_read -= nbytes

        bitmap_image = buf
        img = Image.open(BytesIO(bitmap_image))
        img_array = np.asarray(img.convert('RGB'))

        img_array = np.flip(img_array, axis=2)
        return img_array

    def close(self):
        self.send_sock.shutdown(socket.SHUT_WR)
        self.send_sock.close()

        self.recv_sock.shutdown(socket.SHUT_WR)
        self.recv_sock.close()
