"""
This test needs to be run from WITHIN the tet directory!!!
"""

import unittest
import logging

from pyrocko import util
from pyrocko.trace import snuffle

from pinky.data import SourceConfig, SynthesizerData
from pinky.config import PinkyConfig

from beat.heart import ArrivalTaper, Filter


store_superdirs = '/home/gesa/MTinv_tests/gfdbs/'
fn_stations = 'data/meta/stations/ZS_pyrocko.pf'


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
            channels=['N', 'E', 'Z'],
            center_sources=True)   

    def test_synthesizer_init(self):
        self.sc.source_config.set_ranges()
        print(self.sc.source_config.get_uniform_random())
        self.sc.update_source_randomly       
        print(self.sc)

    def test_synthesizer_setup(self):
        self.pc = PinkyConfig(
            data_generator=self.sc,
            fn_stations=fn_stations,
            n_classes=8,
            reference_station='ZS.D085.'
            )
        self.sc.set_config(self.pc)
        arr_trcs, label = next(self.sc.iter_examples_and_labels())
        #snuffle(trcs)


if __name__ == "__main__":
    util.setup_logging('test_synthesizer', 'warning')
    unittest.main()