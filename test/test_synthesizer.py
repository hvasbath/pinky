import unittest
import logging

from pyrocko import util
from pinky.data import SourceConfig
from pinky.data import SynthesizerData
from beat.heart import ArrivalTaper, Filter

store_superdirs = '/home/gesa/MTinv_tests/gfdbs/'

class TestSynthesizer(unittest.TestCase):

    def setUp(self):
        self.source_config = SourceConfig()
        self.arrival_taper = ArrivalTaper(a=-40., b=-30., c=160., d=170.)
        self.filterer = Filter(
            lower_corner=0.01,
            upper_corner=0.1,
            order=3)
        self.sc = SynthesizerData(
            taperer=self.arrival_taper,
            filterer=self.filterer,
            wavename='any_P',
            store_id='crust2_m5',
            store_superdirs=store_superdirs,
            channels=['N', 'E', 'Z'])

    def test_synthesizer_init(self):
        print(self.sc)


if __name__ == "__main__":
    util.setup_logging('test_synthesizer', 'info')
    unittest.main()