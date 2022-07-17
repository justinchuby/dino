from dino import modules
import unittest

class VisionTransformerTest(unittest.TestCase):
    def test_vit_tiny_can_initilize(self):
        model = modules.vit_tiny()
        self.assertIsNotNone(model)
