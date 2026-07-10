import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ImportBoundaryTests(unittest.TestCase):
    def test_package_imports_without_optional_ml_dependencies(self):
        import drone_event_nav

        self.assertIsInstance(drone_event_nav.__version__, str)
        self.assertTrue(drone_event_nav.__version__)
        for optional_module in ("torch", "transformers", "numpy"):
            self.assertNotIn(optional_module, sys.modules)


if __name__ == "__main__":
    unittest.main()
