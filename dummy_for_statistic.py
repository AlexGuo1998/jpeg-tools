import typing


# For gathering statistics, will not write anything!


class DummyHuffmanTree:
    def __init__(self):
        self.stat = {}

    # return (length, code)
    def encode(self, symbol) -> typing.Tuple[int, int]:
        self.stat[symbol] = self.stat.get(symbol, 0) + 1
        return 0, 0


class DummyWriter:
    def write(self, length, val):
        pass
