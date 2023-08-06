import struct
import typing

import dummy_for_statistic
from bit_helper import BitReader, BitWriter
from huffman import HuffTree
from jpeg_markers import *


class JpegImage:
    def __init__(self):
        self.content: typing.List[typing.Union[Marker, bytes]] = []
        '''marker or scan_data. scan_data has FF00 replaced by FF.'''

        self.dct_data: typing.List[typing.List[typing.List[typing.List[int]]]] = []
        '''dct_data[channel][block_y][block_x] = 8x8 block in DCT coefficients (already quantized) (len=64, in zigzag order)
        channel: component ID `Ci` is `SOFn.components[channel].c`
        '''

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

    @staticmethod
    def _prepare_decode_sof(marker):
        dct_data_shape = []
        component_layout = []
        if marker.sof_type != 0:
            raise NotImplementedError('Only SOF0 decoding implemented')
        h_max = max(h for (c, h, v, tq) in marker.components)
        v_max = max(v for (c, h, v, tq) in marker.components)
        x_block = -(-marker.x // (h_max * 8))
        y_block = -(-marker.y // (v_max * 8))
        for (c, h, v, tq) in marker.components:
            x = x_block * h
            y = y_block * v
            dct_data_shape.append((y, x))
            component_layout.append((c, h, v))
        return component_layout, x_block, y_block, dct_data_shape

    @staticmethod
    def _prepare_decode_sos(marker, component_layout):
        decode_layout = []
        for (cs, td, ta) in marker.components:
            channel, h, v = next(
                (i, h, v)
                for i, (ci, h, v) in enumerate(component_layout)
                if ci == cs)
            for dy in range(v):
                for dx in range(h):
                    decode_layout.append((channel, td, ta, dx, dy))
        return decode_layout

    @staticmethod
    def _decode_block(
            dc_table: HuffTree,
            ac_table: HuffTree,
            reader: BitReader,
    ) -> typing.List[int]:

        def read_coefficient(ssss):
            nonlocal reader
            if ssss == 0:
                return 0
            val = reader.read(ssss)  # read ssss bits
            if val & (1 << (ssss - 1)):
                return val  # >xxx: face value
            return val - (1 << ssss) + 1  # else: subtract

        out = [0] * 64

        # DC
        l, ssss = dc_table.decode(reader.peek(16))
        reader.skip(l)
        out[0] = read_coefficient(ssss)

        # AC
        index = 0
        while index < 63:  # will write at >=(index+1)
            l, rrrr_ssss = ac_table.decode(reader.peek(16))
            reader.skip(l)
            rrrr, ssss = divmod(rrrr_ssss, 0x10)
            if ssss:
                index += rrrr + 1
                out[index] = read_coefficient(ssss)
            elif rrrr == 0:
                break  # EOB
            elif rrrr == 15:
                index += 16  # ZRL: skip 16
            else:
                raise ValueError(f'Bad AC code 0x{rrrr_ssss:02X}')

        return out

    @staticmethod
    def _encode_block(
            block: typing.List[int],
            dc_table: HuffTree,
            ac_table: HuffTree,
            writer: BitWriter,
    ) -> None:
        def write_coefficient(ssss, val):
            nonlocal writer
            if ssss == 0:
                return
            if val < 0:
                val += (1 << ssss) - 1
            writer.write(ssss, val)

        # DC
        ssss = block[0].bit_length()
        l, code = dc_table.encode(ssss)
        writer.write(l, code >> (16 - l))
        write_coefficient(ssss, block[0])

        # AC
        last_write_index = 0
        for index in range(1, 64):
            val = block[index]
            if val == 0:
                continue
            rrrr = index - last_write_index - 1
            while rrrr > 15:
                rrrr -= 16
                l, code = ac_table.encode(0xF0)  # ZRL
                writer.write(l, code >> (16 - l))
            ssss = val.bit_length()
            l, code = ac_table.encode((rrrr << 4) | ssss)
            writer.write(l, code >> (16 - l))
            write_coefficient(ssss, val)
            last_write_index = index
        if last_write_index != 63:
            l, code = ac_table.encode(0x00)  # ZRL
            writer.write(l, code >> (16 - l))

    # huffman decode bitstream to self.dct_data
    def decode_dct_data(self) -> None:
        # (tc, th) -> table
        huffman_tables: typing.Dict[typing.Tuple[int, int], HuffTree] = {}
        # [(ci, h, v)]
        component_layout: typing.List[typing.Tuple[int, int, int]] = []
        # (channel[idx of component_layout], td, ta, offset_x, offset_y)
        decode_layout = []
        x_block = y_block = 0

        for marker in self.content:
            if isinstance(marker, DHTMarker):
                for tc, th, v in marker.tables:
                    huffman_tables[(tc, th)] = HuffTree.from_length_symbol(v)
            if isinstance(marker, SOFnMarker):
                component_layout, x_block, y_block, dct_data_shape = self._prepare_decode_sof(marker)
                # dct_data[0..y][0..x] = []
                self.dct_data = [[[[] for _ in range(x)] for _ in range(y)]
                                 for (y, x) in dct_data_shape]
            if isinstance(marker, SOSMarker):
                decode_layout = self._prepare_decode_sos(marker, component_layout)
            if isinstance(marker, bytes):
                # decode now!
                reader = BitReader(marker + b'\xFF\xFF', 0)  # avoid over-read
                component_last_dc = [0] * len(component_layout)
                for yb in range(y_block):
                    for xb in range(x_block):
                        for channel, td, ta, dx, dy in decode_layout:
                            ci, h, v = component_layout[channel]
                            x = xb * h + dx
                            y = yb * v + dy
                            block = self._decode_block(
                                huffman_tables[(0, td)],
                                huffman_tables[(1, ta)],
                                reader)
                            component_last_dc[channel] = block[0] = (
                                    component_last_dc[channel] + block[0])
                            self.dct_data[channel][y][x] = block

    # huffman encode to bitstream, with self.dct_data
    # gather_huffman_info_only: do not encode, rather return {(tc, th): {symbol: weight}}
    def encode_dct_data(self, gather_huffman_info_only=False):
        # (tc, th) -> table
        huffman_tables: typing.Dict[typing.Tuple[int, int], HuffTree] = {}
        # [(ci, h, v)]
        component_layout: typing.List[typing.Tuple[int, int, int]] = []
        # (channel[idx of component_layout], td, ta, offset_x, offset_y)
        decode_layout = []
        x_block = y_block = 0

        for marker_index, marker in enumerate(self.content):
            if isinstance(marker, DHTMarker):
                for tc, th, v in marker.tables:
                    if gather_huffman_info_only:
                        table = dummy_for_statistic.DummyHuffmanTree()
                    else:
                        table = HuffTree.from_length_symbol(v)
                    huffman_tables[(tc, th)] = table
            if isinstance(marker, SOFnMarker):
                component_layout, x_block, y_block, _ = self._prepare_decode_sof(marker)
            if isinstance(marker, SOSMarker):
                decode_layout = self._prepare_decode_sos(marker, component_layout)
            if isinstance(marker, bytes):
                # encode now!
                if gather_huffman_info_only:
                    writer = dummy_for_statistic.DummyWriter()
                else:
                    writer = BitWriter(0)
                component_last_dc = [0] * len(component_layout)
                for yb in range(y_block):
                    for xb in range(x_block):
                        for channel, td, ta, dx, dy in decode_layout:
                            ci, h, v = component_layout[channel]
                            x = xb * h + dx
                            y = yb * v + dy
                            block = self.dct_data[channel][y][x][:]
                            component_last_dc[channel], block[0] = block[0], block[0] - component_last_dc[channel]
                            self._encode_block(
                                block,
                                huffman_tables[(0, td)],
                                huffman_tables[(1, ta)],
                                writer)
                # write back
                if not gather_huffman_info_only:
                    # pad unused bits with FF
                    writer.write(8, 0xFF)
                    self.content[marker_index] = writer.get_data()[:-1]
        if gather_huffman_info_only:
            return {k: v.stat for k, v in huffman_tables.items()}

    # dequantize and DCT decode self.dct_data to self.pixel_data
    def decode_pixel_data(self) -> None:
        raise NotImplementedError

    # DCT encode and quantize to self.dct_data, with self.pixel_data
    def encode_pixel_data(self) -> None:
        raise NotImplementedError


def optimize_table(im, option=HuffTree.GenerateOption.sort_same_length_by_code):
    huffman_info = im.encode_dct_data(gather_huffman_info_only=True)
    for target_tc_th, code_weight_dict in huffman_info.items():
        # generate optimal table
        tree = HuffTree.generate_optimal(
            code_weight_dict.items(),
            max_length=16,
            jpeg_compatible=True,
            option=option)
        v = tree.to_length_symbol()

        # update data
        for marker_index, marker in enumerate(im.content):
            if isinstance(marker, DHTMarker):
                for i, (tc, th, _) in enumerate(marker.tables):
                    if (tc, th) == target_tc_th:
                        # updated!
                        marker.tables[i] = (tc, th, v)
                        # print(f'Updated {tc}-{th}')

    # serialize to bytes
    for marker_index, marker in enumerate(im.content):
        if isinstance(marker, DHTMarker):
            marker.update_data()


def decrypt_comibushi(filename, filename_out):
    with open(filename, 'rb') as f:
        data = f.read()
    im = JpegImage()
    im.load(data)

    im.decode_dct_data()

    # restore the data
    for ch in im.dct_data:
        y = len(ch)
        x = len(ch[0])
        y_shuffle = y // 4
        x_shuffle = x // 4

        new = [[[] for _ in range(x)] for _ in range(y)]
        for yy in range(y):
            y1, y2 = divmod(yy, y_shuffle)
            for xx in range(x):
                x1, x2 = divmod(xx, x_shuffle)
                # only process 0123
                if y1 >= 4 or x1 >= 4:
                    new[yy][xx] = ch[yy][xx]
                else:
                    new[yy][xx] = ch[x1 * y_shuffle + y2][y1 * x_shuffle + x2]

        ch[:] = new

    optimize_table(im)
    im.encode_dct_data()

    with open(filename_out, 'wb') as f:
        im.dump_file(f)


def test():
    filename = 'test.jpg'
    with open(filename, 'rb') as f:
        data = f.read()
    im = JpegImage()
    im.load(data)
    print(im.content_readable)

    im.decode_dct_data()
    info = im.encode_dct_data(gather_huffman_info_only=True)
    print()
    print('Huffman stats:')
    for (tc, th), v in sorted(info.items()):
        print(f'  table {tc}-{th} weights: {v}')
    im.encode_dct_data()

    # with open('reconstructed.jpg', 'wb') as f:
    #     im.dump_file(f)
    reconstructed = im.dump()
    assert reconstructed == data


if __name__ == '__main__':
    test()
