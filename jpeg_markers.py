import struct
import typing

__all__ = [
    'Marker', 'DQTMarker', 'DHTMarker', 'SOFnMarker', 'SOSMarker',
    'DRIMarker', 'SOIMarker', 'EOIMarker', 'RSTnMarker'
]


class Marker:
    MARKER_NAMES = {
        b'\xFF\xC0': ('SOF0', 'Baseline DCT'),
        b'\xFF\xC1': ('SOF1', 'Extended sequential DCT'),
        b'\xFF\xC2': ('SOF2', 'Progressive DCT'),
        b'\xFF\xC3': ('SOF3', 'Lossless'),

        b'\xFF\xC5': ('SOF5', 'Differential sequential DCT'),
        b'\xFF\xC6': ('SOF6', 'Differential progressive DCT'),
        b'\xFF\xC7': ('SOF7', 'Differential lossless'),

        b'\xFF\xC8': ('JPG', 'Future extension'),
        b'\xFF\xC9': ('SOF9', 'Extended sequential DCT (arithmetic)'),
        b'\xFF\xCA': ('SOF10', 'Progressive DCT (arithmetic)'),
        b'\xFF\xCB': ('SOF11', 'Lossless (arithmetic)'),

        b'\xFF\xCD': ('SOF13', 'Differential sequential DCT (arithmetic)'),
        b'\xFF\xCE': ('SOF14', 'Differential progressive DCT (arithmetic)'),
        b'\xFF\xCF': ('SOF15', 'Differential lossless (arithmetic)'),

        b'\xFF\xC4': ('DHT', 'Define huffman table'),
        b'\xFF\xCC': ('DAC', 'Define arithmetic coding conditioning'),

        b'\xFF\xD0': ('RST0', 'Restart 0'),
        b'\xFF\xD1': ('RST1', 'Restart 1'),
        b'\xFF\xD2': ('RST2', 'Restart 2'),
        b'\xFF\xD3': ('RST3', 'Restart 3'),
        b'\xFF\xD4': ('RST4', 'Restart 4'),
        b'\xFF\xD5': ('RST5', 'Restart 5'),
        b'\xFF\xD6': ('RST6', 'Restart 6'),
        b'\xFF\xD7': ('RST7', 'Restart 7'),

        b'\xFF\xD8': ('SOI', 'Start of image'),
        b'\xFF\xD9': ('EOI', 'End of image'),
        b'\xFF\xDA': ('SOS', 'Start of scan'),
        b'\xFF\xDB': ('DQT', 'Define quantization table'),
        b'\xFF\xDC': ('DNL', 'Define number of lines'),
        b'\xFF\xDD': ('DRI', 'Define restart interval'),
        b'\xFF\xDE': ('DHP', 'Define hierarchical progression'),
        b'\xFF\xDF': ('EXP', 'Expand reference component'),

        b'\xFF\xE0': ('APP0', 'Application 0'),
        b'\xFF\xE1': ('APP1', 'Application 1'),
        b'\xFF\xE2': ('APP2', 'Application 2'),
        b'\xFF\xE3': ('APP3', 'Application 3'),
        b'\xFF\xE4': ('APP4', 'Application 4'),
        b'\xFF\xE5': ('APP5', 'Application 5'),
        b'\xFF\xE6': ('APP6', 'Application 6'),
        b'\xFF\xE7': ('APP7', 'Application 7'),
        b'\xFF\xE8': ('APP8', 'Application 8'),
        b'\xFF\xE9': ('APP9', 'Application 9'),
        b'\xFF\xEA': ('APP10', 'Application 10'),
        b'\xFF\xEB': ('APP11', 'Application 11'),
        b'\xFF\xEC': ('APP12', 'Application 12'),
        b'\xFF\xED': ('APP13', 'Application 13'),
        b'\xFF\xEE': ('APP14', 'Application 14'),
        b'\xFF\xEF': ('APP15', 'Application 15'),

        b'\xFF\xF0': ('JPG0', 'Reserved JPEG 0'),
        b'\xFF\xF1': ('JPG1', 'Reserved JPEG 1'),
        b'\xFF\xF2': ('JPG2', 'Reserved JPEG 2'),
        b'\xFF\xF3': ('JPG3', 'Reserved JPEG 3'),
        b'\xFF\xF4': ('JPG4', 'Reserved JPEG 4'),
        b'\xFF\xF5': ('JPG5', 'Reserved JPEG 5'),
        b'\xFF\xF6': ('JPG6', 'Reserved JPEG 6'),
        b'\xFF\xF7': ('JPG7', 'Reserved JPEG 7'),
        b'\xFF\xF8': ('JPG8', 'Reserved JPEG 8'),
        b'\xFF\xF9': ('JPG9', 'Reserved JPEG 9'),
        b'\xFF\xFA': ('JPG10', 'Reserved JPEG 10'),
        b'\xFF\xFB': ('JPG11', 'Reserved JPEG 11'),
        b'\xFF\xFC': ('JPG12', 'Reserved JPEG 12'),
        b'\xFF\xFD': ('JPG13', 'Reserved JPEG 13'),

        b'\xFF\xFE': ('COM', 'Comment'),
    }

    # raw markers without length & data
    MARKER_WITHOUT_DATA = {
        b'\xFF\xD0',  # ('RST0', 'Restart 0 (mod 8)'),
        b'\xFF\xD1',  # ('RST1', 'Restart 1 (mod 8)'),
        b'\xFF\xD2',  # ('RST2', 'Restart 2 (mod 8)'),
        b'\xFF\xD3',  # ('RST3', 'Restart 3 (mod 8)'),
        b'\xFF\xD4',  # ('RST4', 'Restart 4 (mod 8)'),
        b'\xFF\xD5',  # ('RST5', 'Restart 5 (mod 8)'),
        b'\xFF\xD6',  # ('RST6', 'Restart 6 (mod 8)'),
        b'\xFF\xD7',  # ('RST7', 'Restart 7 (mod 8)'),

        b'\xFF\xD8',  # ('SOI', 'Start of image'),
        b'\xFF\xD9',  # ('EOI', 'End of image'),
    }

    REGISTERED_CLASSES: typing.Dict[bytes, typing.Type['Marker']] = {}

    def __init__(self, marker: bytes, data: typing.Optional[bytes]):
        assert marker[0] == 0xFF
        self.marker: bytes = marker
        self.data: typing.Optional[bytes] = data

    def update_data(self):
        pass

    @property
    def marker_name_description(self) -> typing.Tuple[str, str]:
        return self.MARKER_NAMES.get(self.marker, ('???', 'Unknown marker'))

    @classmethod
    def register(cls, cls1: typing.Type['Marker'], marker: bytes):
        cls.REGISTERED_CLASSES[marker] = cls1

    @classmethod
    def parse(cls, raw: bytes, pos: int) -> typing.Tuple['Marker', int]:
        marker = raw[pos:pos + 2]
        if marker in cls.REGISTERED_CLASSES:
            new_class = cls.REGISTERED_CLASSES[marker]
            return new_class.parse(raw, pos)
        # manual parsing
        if marker in cls.MARKER_WITHOUT_DATA:
            return cls(marker, None), 2
        else:
            length, = struct.unpack('>H', raw[pos + 2:pos + 4])
            return cls(marker, raw[pos + 4:pos + 2 + length]), 2 + length

    def __str__(self):
        MAX_DATA_PREVIEW_LEN = 32
        name, desc = self.marker_name_description
        if self.data is None:
            return f'{name} ({desc})'
        elif len(self.data) <= MAX_DATA_PREVIEW_LEN:
            return f'{name} ({desc})\n  {self.data}'
        else:
            return f'{name} ({desc})\n  {self.data[:MAX_DATA_PREVIEW_LEN]}... (len={len(self.data)})'

    def __init_subclass__(cls, *, register_marker: typing.Union[bytes, typing.Iterable[bytes], None] = None):
        if register_marker is not None:
            if isinstance(register_marker, bytes):
                register_marker = (register_marker,)
            for m in register_marker:
                cls.register(cls, m)
        # clear registered_classes, avoid looping
        cls.REGISTERED_CLASSES = {}


class DQTMarker(Marker, register_marker=b'\xFF\xDB'):
    def __init__(self, marker: bytes, data: bytes):
        super(DQTMarker, self).__init__(marker, data)
        i = 0
        self.tables = []
        while True:
            if i == len(self.data):
                return
            pq_tq, = struct.unpack('>B', self.data[i:i + 1])
            i += 1
            pq, tq = divmod(pq_tq, 16)
            if pq == 0:
                # 8-bit
                table = struct.unpack('>64B', self.data[i:i + 64])
                i += 64
            elif pq == 1:
                # 16-bit
                table = struct.unpack('>64H', self.data[i:i + 128])
                i += 128
            else:
                raise ValueError(f'Invalid Pq ({pq}) in DQT')
            self.tables.append((pq, tq, table))

    def update_data(self):
        raise NotImplementedError

    def __str__(self):
        name, desc = self.marker_name_description
        tables = []
        for pq, tq, t in self.tables:
            precision = '8' if pq == 0 else '16'
            tables.append(f'  Table {tq} ({precision}-bit):\n'
                          f'    {t}')
        tables_str = '\n'.join(tables)
        return f'{name} ({desc})\n{tables_str}'


class DHTMarker(Marker, register_marker=b'\xFF\xC4'):
    def __init__(self, marker: bytes, data: bytes):
        super(DHTMarker, self).__init__(marker, data)
        self.tables = []
        i = 0
        while True:
            if i == len(self.data):
                return
            tc_th, *lengths = struct.unpack('>B16B', self.data[i:i + 17])
            i += 17
            tc, th = divmod(tc_th, 16)
            assert tc in (0, 1), f'Invalid Tc ({tc}) in DHT'
            v = []
            for li in lengths:
                if li == 0:
                    vi = ()
                else:
                    vi = struct.unpack('>' + 'B' * li, self.data[i:i + li])
                i += li
                v.append(vi)
            self.tables.append((tc, th, v))

    def update_data(self):
        raise NotImplementedError

    def __str__(self):
        name, desc = self.marker_name_description
        tables = []
        for tc, th, v in self.tables:
            v_str = '\n'.join(f'    {x}' for x in v)
            dc_ac = 'DC' if tc == 0 else 'AC'
            tables.append(f'  Table {dc_ac}-{th}:\n'
                          f'{v_str}')
        tables_str = '\n'.join(tables)
        return f'{name} ({desc})\n{tables_str}'


class SOFnMarker(Marker,
                 register_marker=(b'\xFF\xC0', b'\xFF\xC1', b'\xFF\xC2', b'\xFF\xC3',
                                  b'\xFF\xC9', b'\xFF\xCA', b'\xFF\xCB')):
    def __init__(self, marker: bytes, data: bytes):
        super(SOFnMarker, self).__init__(marker, data)
        self.p, self.y, self.x, nf = struct.unpack('>BHHB', self.data[0:6])
        i = 6
        self.components = []
        for m in range(nf):
            c, h_v, tq = struct.unpack('>BBB', self.data[i:i + 3])
            h, v = divmod(h_v, 16)
            self.components.append((c, h, v, tq))
            i += 3
        assert i == len(self.data), 'Unknown padding data'

    def update_data(self):
        raise NotImplementedError

    @property
    def sof_type(self):
        return self.marker[1] & 0x0F

    def __str__(self):
        name, desc = self.marker_name_description
        components = '\n'.join(f'  Component {i} (id #{c}): {h}x{v} in a block, quantize table {tq}'
                               for i, (c, h, v, tq) in enumerate(self.components, start=1))
        return (
            f'{name} ({desc})\n'
            f'  Precision: {self.p}-bit\n'
            f'  Size: {self.x}x{self.y}\n'
            f'{components}'
        )


class SOSMarker(Marker, register_marker=b'\xFF\xDA'):
    def __init__(self, marker: bytes, data: bytes):
        super(SOSMarker, self).__init__(marker, data)
        ns, = struct.unpack('>B', self.data[0:1])
        i = 1
        self.components = []
        for m in range(ns):
            cs, td_ta = struct.unpack('>BB', self.data[i:i + 2])
            td, ta = divmod(td_ta, 16)
            self.components.append((cs, td, ta))
            i += 2
        self.ss, self.se, ah_al = struct.unpack('>BBB', self.data[i:i + 3])
        self.ah, self.al = divmod(ah_al, 16)
        assert i + 3 == len(self.data), 'Unknown padding data'

    def update_data(self):
        raise NotImplementedError

    def __str__(self):
        name, desc = self.marker_name_description
        components = '\n'.join(f'  Component {i} (id #{cs}) Huff table: DC-{td}, AC-{ta}'
                               for i, (cs, td, ta) in enumerate(self.components, start=1))
        return (
            f'{name} ({desc})\n'
            f'{components}\n'
            f'  Scan range: {self.ss}-{self.se}\n'
            f'  Ah={self.ah} Al={self.al}'
        )


class DRIMarker(Marker, register_marker=b'\xFF\xDD'):
    @property
    def restart_interval(self):
        return struct.unpack('>H', self.data)[0]

    @restart_interval.setter
    def restart_interval(self, value):
        self.data = struct.pack('>H', value)

    def __str__(self):
        name, desc = self.marker_name_description
        return f'{name} ({desc})\n  Restart interval: {self.restart_interval}'


class SOIMarker(Marker, register_marker=b'\xFF\xD8'):
    pass


class EOIMarker(Marker, register_marker=b'\xFF\xD9'):
    pass


class RSTnMarker(Marker,
                 register_marker=(b'\xFF\xD0', b'\xFF\xD1', b'\xFF\xD2', b'\xFF\xD3',
                                  b'\xFF\xD4', b'\xFF\xD5', b'\xFF\xD6', b'\xFF\xD7')):
    def rst_type(self):
        return self.marker[1] & 0x0F
