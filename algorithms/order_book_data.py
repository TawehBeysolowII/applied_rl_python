#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:56:09 2019

@author: tawehbeysolow
"""

from tgym.core import DataGenerator
import pandas as pan

class bid_ask_data(DataGenerator):
    
    def __init__(self, **gen_kwargs):
        """Initialisation function. The API (gen_kwargs) should be defined in
        the function _generator.
        """
        self._trainable = False
        self.gen_kwargs = gen_kwargs
        DataGenerator.rewind(self)
        self.n_products = len(self.next()) / 2
        DataGenerator.rewind(self)
        
    @staticmethod
    def _generator():
        columns = ['bid', 'ask']
        raw_data = pan.read_csv('/Users/tawehbeysolow/Downloads/amazon_order_book_data.csv', names=columns)
        scaled_data = raw_data/float(1e5)
    
        for i in range(0, len(scaled_data)):            
            yield scaled_data.ix[i, 0], scaled_data.ix[i, 1]
            
    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        print "End of data reached, rewinding."
        super(self.__class__, self).rewind()
    
    
    def next(self):
        """Return the next element in the generator.
        Args:
            numpy.array: next row of the generator
        """
        try:
            return next(self.generator)
        except StopIteration as e:
            self._iterator_end()
            raise(e)

    def rewind(self):
        """Rewind the generator.
        """
        self.generator = self._generator(**self.gen_kwargs)


if __name__ == '__main__':


    generator = bid_ask_data(filename='amazon_order_book_data.csv', filepath='/Users/tawehbeysolow/Downloads/')
    prices_time_series = [next(generator.preprocess()) for _ in range(100)]
    import pdb; pdb.set_trace()
