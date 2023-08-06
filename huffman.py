import bisect
import enum
import itertools
import operator
import typing
import unittest

VT = typing.TypeVar('VT')

_missing = object()


class HuffTree(typing.Generic[VT]):
    class GenerateOption(enum.Enum):
        sort_same_length_by_probability = 0
        sort_same_length_by_code = 1
        sort_same_length_by_original_order = 2

    def __init__(
            self,
            code_to_symbol: typing.List[typing.Dict[int, VT]],
            symbol_to_code: typing.Dict[VT, typing.Tuple[int, int]],
            symbol_shift: int,
    ):
        # (1<<symbol_shift)-1 mask for the symbol
        self.symbol_shift = symbol_shift

        # mask for 1-length code
        self._first_mask = ~((1 << (symbol_shift - 1)) - 1)

        # cts[len][code] -> symbol, for decode
        self.code_to_symbol = code_to_symbol

        # stc[symbol] -> (len, code), for decode
        self.symbol_to_code = symbol_to_code

    # return (consumed_length, symbol)
    def decode(self, code: int) -> typing.Tuple[int, VT]:
        mask = self._first_mask
        for consumed_length, d in enumerate(self.code_to_symbol, 1):
            sym = d.get(code & mask, _missing)
            if sym is not _missing:
                return consumed_length, sym
            mask >>= 1
        raise KeyError(f'Can not decode code 0x{code:0{self.symbol_shift // 8}X}')

    # return (length, code)
    def encode(self, symbol: VT) -> typing.Tuple[int, int]:
        return self.symbol_to_code[symbol]

    @classmethod
    def generate_optimal(
            cls,
            symbol_weight: typing.Iterable[typing.Tuple[VT, int]],
            max_length: int = 16,
            jpeg_compatible: bool = False,
            option: GenerateOption = GenerateOption.sort_same_length_by_probability,
    ):
        if option == HuffTree.GenerateOption.sort_same_length_by_probability:
            symbol_weight_l = sorted(symbol_weight, key=operator.itemgetter(1), reverse=True)
            weight_presorted = True
        elif option == HuffTree.GenerateOption.sort_same_length_by_code:
            symbol_weight_l = sorted(symbol_weight, key=operator.itemgetter(0))
            weight_presorted = False
        elif option == HuffTree.GenerateOption.sort_same_length_by_original_order:
            symbol_weight_l = list(symbol_weight)
            weight_presorted = False
        else:
            raise ValueError(f'Unknown option: {option!r}')

        lengths = cls.generate_code_length(
            (weight for symbol, weight in symbol_weight_l),
            max_length, jpeg_compatible, weight_presorted=weight_presorted)

        grouped_by_length = tuple([] for _ in range(max_length))
        for (symbol, _), length in zip(symbol_weight_l, lengths):
            grouped_by_length[length - 1].append(symbol)
        return cls.from_length_symbol(grouped_by_length)

    @classmethod
    def from_length_symbol(cls, grouped_by_length: typing.Sequence[typing.Sequence[VT]]):
        max_length = len(grouped_by_length)

        code_to_symbol = [{} for _ in grouped_by_length]
        max_length_aligned_to_byte = -(-max_length // 8) * 8
        symbol_to_code = {}
        curr_code = 0
        for length, symbols in enumerate(grouped_by_length, 1):
            for symbol in symbols:
                # curr_code_shifted = curr_code >> (max_length - length)
                # print(f'sym={symbol:5}, code={curr_code_shifted:0{length}b}')

                code_to_symbol[length - 1][curr_code] = symbol
                symbol_to_code[symbol] = (length, curr_code)

                curr_code += 1 << (max_length_aligned_to_byte - length)

        assert curr_code <= (1 << max_length_aligned_to_byte), 'sanity check'

        return cls(code_to_symbol, symbol_to_code, max_length_aligned_to_byte)

    def to_length_symbol(self) -> typing.Tuple[typing.Tuple[VT, ...], ...]:
        return tuple(tuple(symbol for (code, symbol) in sorted(s.items()))
                     for s in self.code_to_symbol)

    @staticmethod
    def generate_code_length(
            weights: typing.Iterable[int],
            max_length: int = 16,
            jpeg_compatible: bool = False,
            *,
            weight_presorted: bool = False,
    ) -> typing.Iterator[int]:
        # jpeg_compatible: reserve all-1-bits code. The number of codes in the last layer is odd
        # weight_presorted: weight is pre-sorted in descending order
        # return: bit-lengths for each weight

        # (index, weight), weight descending
        if weight_presorted:
            weight_sorted: typing.List[typing.Tuple[int, int]] = list(enumerate(weights))
        else:
            weight_sorted: typing.List[typing.Tuple[int, int]] = sorted(
                enumerate(weights), key=operator.itemgetter(1), reverse=True)
        total_items = len(weight_sorted)

        # jpeg_compatible: The least-weight item must be paired (with a dummy item) and moved to the upper layer
        # Also we shift every `weight` in the paper left `max_length` bits, and shift it right every layer
        if jpeg_compatible:
            total_width_shifted = total_items << max_length  # add 1 dummy item
            least_id, least_weight = weight_sorted.pop(-1)
        else:
            total_width_shifted = (total_items - 1) << max_length
            least_id = least_weight = None

        # (weight, {index: max_layer})
        packet_type = typing.Tuple[int, typing.Dict[int, int]]

        # We initialize packets for each layer (l), only when we are dealing with that layer
        packets: typing.List[packet_type] = []  # where width==2**(-l)
        packets_next: typing.List[packet_type] = []  # where width==2**(-l+1)

        layer = max_length
        chosen_children = {}  # the final result

        def merge_children(c1, c2):
            # keep largest value
            for k, v in c2.items():
                c1[k] = max(c1.get(k, 0), v)

        while total_width_shifted:
            if layer > 0:
                # add items for this layer
                assert not (total_width_shifted & 1)
                if jpeg_compatible:
                    # the least-weight item is already paired
                    packets_next = [(least_weight, {least_id: layer})]

                # TODO: left or right?
                # This shouldn't affect the final weight sum, but the tree shape can be different
                packets.extend((weight, {id_: layer}) for id_, weight in weight_sorted)
                # or:
                # packets = [(weight, {id_: layer}) for id_, weight in weight_sorted] + packets

                packets.sort(key=operator.itemgetter(0), reverse=True)
            else:
                # layer <= 0, check if total_width has a 1-bit in that position
                if total_width_shifted & 1:
                    # the least-weight item is chosen
                    # throw an exception (bug situation) if `packets` is empty
                    merge_children(chosen_children, packets.pop(-1)[1])
            layer -= 1

            # merge the packets, until only <=1 packet left
            while len(packets) >= 2:
                w1, c1 = packets.pop(-1)
                w2, c2 = packets.pop(-1)
                merge_children(c1, c2)
                packets_next.append((w1 + w2, c1))

            # next layer
            total_width_shifted >>= 1
            packets_next.reverse()  # keep descending order for better sorting (we inserted as ascending order)
            packets = packets_next
            packets_next = []

        return (chosen_children[i] for i in range(total_items))


class HuffTreeGeneratorTest(unittest.TestCase):
    RANDOM_TEST_ITERS = 500

    @staticmethod
    def _make_data():
        import random
        import math
        weights = [random.randint(1, 10000) for _ in range(random.randint(10, 500))]
        max_depth = random.randint(int(math.log2(len(weights))) + 2, 16)
        return weights, max_depth

    @staticmethod
    def _get_optimal_weight_sum(weights):
        # node_weight, node_weight_sum
        nodes = [(w, 0) for w in sorted(weights)]
        while len(nodes) >= 2:
            (wa, wsa), (wb, wsb) = nodes[:2]
            del nodes[:2]
            bisect.insort(nodes, ((wa + wb), (wsa + wsb + wa + wb)))
        return nodes[0][1]

    def _checkMonotone(self, weights, depths):
        # if the weight is the same, ignore the depth difference (sort by depth ascending)
        sorted_by_weights = sorted(zip(weights, depths), key=lambda x: (-x[0], x[1]))
        depth_last = 1
        for w, d in sorted_by_weights:
            self.assertGreaterEqual(d, depth_last, sorted_by_weights)
            depth_last = d

    def testNonJpeg(self):
        optimal_checked = 0
        capped_checked = 0
        for run in range(self.RANDOM_TEST_ITERS):
            weights, max_depth = self._make_data()
            with self.subTest(run=run, n=len(weights), max_depth=max_depth, weights=weights):
                depths = list(HuffTree.generate_code_length(weights, max_depth, jpeg_compatible=False))

                self.assertEqual(len(weights), len(depths))

                for x in depths:
                    self.assertGreaterEqual(x, 1)
                    self.assertLessEqual(x, max_depth)
                real_max_depth = max(depths)
                must_be_optimal = real_max_depth < max_depth

                count = 0
                max_count = 1
                for layer in range(1, real_max_depth + 1):
                    count *= 2
                    max_count *= 2
                    count += sum(1 for x in depths if x == layer)
                    self.assertLessEqual(count, max_count, f'layer {layer} has invalid node count')
                self.assertEqual(count, max_count, 'suboptimal code: blank detected')
                self._checkMonotone(weights, depths)

                weight_sum = sum(w * d for w, d in zip(weights, depths))
                optimal_weight_sum = self._get_optimal_weight_sum(weights)

                if must_be_optimal:
                    optimal_checked += 1
                    self.assertEqual(weight_sum, optimal_weight_sum, 'not optimal')
                else:
                    capped_checked += 1
                    self.assertGreaterEqual(weight_sum, optimal_weight_sum, 'WTF?')

        self.assertGreater(optimal_checked, 0, 'Did not check for the optimal situation')
        self.assertGreater(capped_checked, 0, 'Did not check for the capped situation')

    def testJpeg(self):
        optimal_checked = 0
        capped_checked = 0
        for run in range(self.RANDOM_TEST_ITERS):
            weights, max_depth = self._make_data()
            with self.subTest(run=run, n=len(weights), max_depth=max_depth, weights=weights):
                depths = list(HuffTree.generate_code_length(weights, max_depth, jpeg_compatible=True))

                self.assertEqual(len(weights), len(depths))

                for x in depths:
                    self.assertGreaterEqual(x, 1)
                    self.assertLessEqual(x, max_depth)
                real_max_depth = max(depths)
                must_be_optimal = real_max_depth < max_depth

                count = 0
                max_count = 1
                for layer in range(1, real_max_depth + 1):
                    count *= 2
                    max_count *= 2
                    count += sum(1 for x in depths if x == layer)
                    self.assertLess(count, max_count, f'layer {layer} node count exceeded')
                self.assertEqual(count, max_count - 1, 'suboptimal code: blank detected')
                self._checkMonotone(weights, depths)

                weight_sum = sum(w * d for w, d in zip(weights, depths))
                optimal_weight_sum = self._get_optimal_weight_sum(weights + [0])

                if must_be_optimal:
                    optimal_checked += 1
                    self.assertEqual(weight_sum, optimal_weight_sum, 'not optimal')
                else:
                    capped_checked += 1
                    self.assertGreaterEqual(weight_sum, optimal_weight_sum, 'WTF?')

        self.assertGreater(optimal_checked, 0, 'Did not check for the optimal situation')
        self.assertGreater(capped_checked, 0, 'Did not check for the capped situation')


if __name__ == '__main__':
    unittest.main()
