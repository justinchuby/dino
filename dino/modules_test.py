import unittest

from dino import modules


class VisionTransformerTest(unittest.TestCase):
    def test_vit_tiny_can_initilize(self):
        model = modules.vit_tiny()
        self.assertIsNotNone(model)
