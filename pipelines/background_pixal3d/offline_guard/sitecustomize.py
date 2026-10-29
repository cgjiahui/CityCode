"""Fail closed if a model loader attempts to download missing assets."""
import socket


def blocked(*args, **kwargs):
    raise RuntimeError('Background Pixal3D is offline: supply all model files locally')


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
