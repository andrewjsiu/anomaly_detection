"""
    Author: Andrew Siu (andrewjsiu@gmail.com)

    -------------------------------------------------
    Detecting Large Purchases within a Social Network
    -------------------------------------------------
    
    This program tests the functions of finding all neighbors of a user
    within 2, 3, 4, or 5 degrees of separation.
    
"""

from network import network
import unittest

class TestNetwork(unittest.TestCase):
    
    def setUp(self):
        self.graph = {'1':{'2'}, '2':{'1','3'}, '3':{'2','4'},
                      '4':{'3','5'}, '5':{'4','6'}, '6':{'5'}}
    
    def test_degree1(self):
        self.assertEqual(network(self.graph, '1', 1), {'2'})
        
    def test_degree2(self):
        self.assertEqual(network(self.graph, '1', 2), {'2','3'})
        
    def test_degree3(self):
        self.assertEqual(network(self.graph, '1', 3), {'2','3','4'}) 
        
    def test_degree4(self):
        self.assertEqual(network(self.graph, '1', 4), {'2','3','4','5'})
        
    def test_degree5(self):
        self.assertEqual(network(self.graph, '1', 5), {'2','3','4','5','6'})
        
if __name__ == '__main__':
    unittest.main(exit=False)

