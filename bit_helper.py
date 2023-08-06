class BitReader:
    def __init__(self, data, initial_bit_position):
        self.data = data
        self.bit_position = initial_bit_position

    def peek(self, length):
        byte_offset, bit_offset = divmod(self.bit_position, 8)
        end_shift = (-bit_offset - length) % 8
        num_bytes = (length + bit_offset + 7) // 8
        data_slice = self.data[byte_offset:byte_offset + num_bytes]
        if len(data_slice) < num_bytes:
            raise ValueError('No more data')
        value = int.from_bytes(data_slice)
        value >>= end_shift
        return value & (1 << length) - 1

    def goto(self, length):
        self.bit_position = length

    def skip(self, length):
        self.bit_position += length

    def read(self, length):
        x = self.peek(length)
        self.skip(length)
        return x


class BitWriter:
    def __init__(self, initial_bit_position):
        self.data = [0]
        self.data_remain = 0
        self.bit_position = initial_bit_position

    def write(self, length, val):
        assert val < (1 << length)
        bytes_written, self.bit_position = divmod(self.bit_position + length, 8)
        bytes_written_8 = bytes_written * 8
        val <<= (8 - self.bit_position)

        # merge the bits to o[-1]
        self.data[-1] |= (val >> bytes_written_8)

        # others
        while bytes_written_8 > 0:
            bytes_written_8 -= 8
            self.data.append((val >> bytes_written_8) & 0xFF)

    def get_data(self):
        if self.bit_position:
            return bytes(self.data)
        else:
            return bytes(self.data[:-1])
