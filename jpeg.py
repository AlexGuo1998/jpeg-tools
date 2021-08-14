import struct
import typing

from jpeg_markers import *


class JpegImage:
    def __init__(self):
        self.content: typing.List[typing.Union[Marker, bytes]] = []
        '''marker or scan_data. scan_data has FF00 replaced by FF.'''

        self.dct_data = []
        '''dct_data[channel][block_y][block_x] = 8x8 block in DCT coefficients (len=64, in zigzag order)'''

        self.pixel_data = []
        '''pixel_data[channel][block_y][block_x] = 8x8 block in pixel (len=64, x then y)'''

    @property
    def content_readable(self) -> str:
        return '\n'.join(
            str(x) if isinstance(x, Marker) else f'---- Payload (escape char removed) size={len(x)}'
            for x in self.content)

    def load(self, file_or_bytes: typing.Union[typing.BinaryIO, bytes]) -> None:
        if isinstance(file_or_bytes, bytes):
            raw = file_or_bytes
        else:
            raw = file_or_bytes.read()
        if raw[0:2] != b'\xFF\xD8':
            raise ValueError('File did not start with SOI, not JPEG image or corrupted?')

        i = 0
        while True:
            try:
                marker, size = Marker.parse(raw, i)
            except Exception as e:
                raise ValueError(f'Error when decoding file offset {i}') from e
            i += size
            self.content.append(marker)
            if isinstance(marker, SOSMarker):
                i = self.__parse_scan_data(raw, i)
            elif isinstance(marker, EOIMarker):
                # decode done
                break

    def __parse_scan_data(self, raw: bytes, i: int):
        start = i
        content_buffer = []
        while True:
            if raw[i] == 0xFF:
                if raw[i + 1] == 0x00:
                    # encounter \xFF\x00, keep \xFF only
                    content_buffer.append(raw[start:i + 1])
                    i += 2
                    start = i
                else:
                    # write current data
                    content_buffer.append(raw[start:i])
                    self.content.append(b''.join(content_buffer))
                    content_buffer = []

                    marker, size = Marker.parse(raw, i)
                    if isinstance(marker, RSTnMarker):
                        # RSTn: continue after parsing
                        self.content.append(marker)
                        i += size
                        start = i
                    else:
                        # any other marker: done parsing scan data
                        return i
            else:
                i += 1

    def dump(self) -> bytes:
        return b''.join(self.dump_generator())

    def dump_file(self, file: typing.BinaryIO) -> None:
        for x in self.dump_generator():
            file.write(x)

    def dump_generator(self) -> typing.Iterable[bytes]:
        for x in self.content:
            if isinstance(x, Marker):
                yield x.marker
                if x.data is not None:
                    data_len = 2 + len(x.data)
                    assert data_len <= 0xFFFF, 'Data too long'
                    yield struct.pack('>H', data_len)
                    yield x.data
            elif isinstance(x, bytes):
                yield x.replace(b'\xFF', b'\xFF\x00')
            else:
                raise ValueError(f'Bad content part: {x}')

    def decode_dct_data(self) -> None:
        raise NotImplementedError

    def encode_dct_data(self) -> None:
        raise NotImplementedError

    def decode_pixel_data(self) -> None:
        raise NotImplementedError

    def encode_pixel_data(self) -> None:
        raise NotImplementedError


def test():
    filename = 'test.jpg'
    with open(filename, 'rb') as f:
        data = f.read()
    im = JpegImage()
    im.load(data)
    print(im.content_readable)
    # with open('reconstructed.jpg', 'wb') as f:
    #     im.dump_file(f)
    reconstructed = im.dump()
    assert reconstructed == data


if __name__ == '__main__':
    test()
