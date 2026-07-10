import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunVerificationMetadataTests(unittest.TestCase):
    def test_result_metadata_uses_effective_movement_parameters(self):
        source_path = REPO_ROOT / "agent_control" / "run_verification.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        parameter_dicts = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
            if "up_step" in keys and "down_step" in keys and "forward_step" in keys:
                parameter_dicts.append(node)

        self.assertEqual(len(parameter_dicts), 1)
        values = {
            key.value: ast.unparse(value)
            for key, value in zip(parameter_dicts[0].keys, parameter_dicts[0].values)
            if isinstance(key, ast.Constant)
        }
        self.assertEqual(values["forward_step"], "movement_params['forward_step']")
        self.assertEqual(values["up_step"], "movement_params['up_step']")
        self.assertEqual(values["down_step"], "movement_params['down_step']")
        self.assertEqual(values["yaw_step"], "movement_params['yaw_step']")


if __name__ == "__main__":
    unittest.main()
