import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AcceleratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global accelerator_report, detect_accelerators, select_device
        from deep_learning_studio.accelerators import (
            accelerator_report,
            detect_accelerators,
            select_device,
        )

    def test_cpu_is_always_available(self):
        accelerators = {item.key: item for item in detect_accelerators()}

        self.assertTrue(accelerators["cpu"].available)
        self.assertEqual(accelerators["cpu"].device.type, "cpu")

    def test_auto_selects_an_available_backend(self):
        device = select_device("auto")
        available_types = {
            item.device.type
            for item in detect_accelerators()
            if item.available and item.device is not None
        }

        self.assertIn(device.type, available_types)

    def test_report_contains_installation_details(self):
        report = accelerator_report()

        self.assertIn("PyTorch:", report)
        self.assertIn("CUDA", report)
        self.assertIn("CPU", report)


if __name__ == "__main__":
    unittest.main()
