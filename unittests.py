import unittest
from pychop import demo_harmonic


class TestClassix(unittest.TestCase):
    
    def check_demo(self):
        checkpoint = 1
            
        try:
            demo_harmonic.main() # test floating point running

        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)


if __name__ == '__main__':
    unittest.main()