import unittest
import pychop
from pychop import demo_harmonic

class TestClassix(unittest.TestCase):
    
    def check_demo(self):
        checkpoint = 1
            
        try:
            pychop.backend('numpy') 
            demo_harmonic.main() # test floating point running
            pychop.backend('torch')
            demo_harmonic.main() # test floating point running
            pychop.backend('jax')
            demo_harmonic.main() # test floating point running
        except:
            checkpoint = 0
        
        self.assertEqual(checkpoint, 1)


if __name__ == '__main__':
    unittest.main()
