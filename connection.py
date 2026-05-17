import socket
import line_packet

class Connection:
    """Wraps socket conn; provides line-based send/recv and raw-audio recv."""
    PACKET_SIZE = 32000 * 5 * 60  # bytes, ~5min @ 16kHz PCM16 mono (~32KB/s)

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.last_line = ""
        self.conn.setblocking(True)

    def send(self, line: str):
        # prevent sending identical line twice
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        return line_packet.receive_lines(self.conn)

    def non_blocking_receive_audio(self):
        try:
            return self.conn.recv(self.PACKET_SIZE)
        except ConnectionResetError:
            return None
