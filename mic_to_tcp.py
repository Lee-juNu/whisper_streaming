import argparse
import socket
import time
import sounddevice as sd
import threading
import line_packet
import numpy as np
import soxr

SAMPLING_RATE = 16000  # Server expects 16kHz mono S16_LE
CHANNELS = 1
DTYPE = 'int16'
# BLOCKSIZE will be computed from input samplerate


def main():
    parser = argparse.ArgumentParser(description="Stream mic audio to TCP as raw 16kHz mono S16_LE")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=43007, help="Server port")
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration to stream (seconds; <=0 runs until interrupted)")
    parser.add_argument("--device", type=str, default=None, help="Input device name or index (optional)")
    parser.add_argument("--samplerate", type=float, default=None, help="Input device samplerate (if omitted, uses device default)")
    args = parser.parse_args()

    # Connect to server
    sock = socket.create_connection((args.host, args.port))
    # Enable non-blocking reads for receiving server transcripts
    sock.setblocking(False)

    # Determine device parameter (int index or name)
    device_param = None
    if args.device is not None:
        if isinstance(args.device, str) and args.device.isdigit():
            device_param = int(args.device)
        else:
            device_param = args.device

    # Determine input samplerate
    input_rate = args.samplerate
    if input_rate is None:
        if device_param is not None:
            try:
                info = sd.query_devices(device_param, kind='input')
                input_rate = float(info.get('default_samplerate', SAMPLING_RATE))
            except Exception:
                input_rate = SAMPLING_RATE
        else:
            input_rate = SAMPLING_RATE

    # Compute blocksize for ~100ms chunks
    BLOCKSIZE = max(1, int(input_rate * 0.1))

    def callback(in_data, frames, time_info, status):
        # in_data: bytes-like buffer in native endian (Windows = little)
        try:
            if input_rate != SAMPLING_RATE:
                # Resample to 16kHz mono int16
                buf = np.frombuffer(in_data, dtype=np.int16)
                x = buf.astype(np.float32) / 32768.0
                y = soxr.resample(x, input_rate, SAMPLING_RATE)
                out = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                sock.sendall(out)
            else:
                sock.sendall(in_data)
        except Exception:
            # Stop stream on send error
            return

    with sd.RawInputStream(
        samplerate=input_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE,
        device=device_param,
    ) as stream:
        # Stream for the requested duration, or indefinitely if seconds <= 0
        try:
            if args.seconds is not None and args.seconds > 0:
                t_end = time.time() + args.seconds
            else:
                t_end = None
            while True:
                # Read audio block
                in_data, _ = stream.read(BLOCKSIZE)
                if input_rate != SAMPLING_RATE:
                    buf = np.frombuffer(in_data, dtype=np.int16)
                    x = buf.astype(np.float32) / 32768.0
                    y = soxr.resample(x, input_rate, SAMPLING_RATE)
                    out = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    sock.sendall(out)
                else:
                    sock.sendall(in_data)

                # Receive and print server transcripts
                try:
                    lines = line_packet.receive_lines(sock)
                except Exception:
                    lines = None
                if lines:
                    for ln in lines:
                        if ln:
                            print(ln)

                time.sleep(0.05)
                if t_end is not None and time.time() >= t_end:
                    break
        except KeyboardInterrupt:
            pass

    sock.close()


if __name__ == "__main__":
    main()
