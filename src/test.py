from test_core import *
from train import ConfigRuntimeController
import argparse
import json
import os
import sys

def test(test_ver="DST_Calib", config_json_path=None, mode="SB"):
    if config_json_path is None:
        raise ValueError("config_json_path must be provided for testing")
    return testAuto(test_ver, config_json_path, mode=mode)

def testAuto(test_ver, config_json_path, mode="SB"):
    rtime_ctrller = ConfigRuntimeController(config_json_path, test_ver=test_ver)
    config_loader = rtime_ctrller.config_loader
    zvdataloader = rtime_ctrller.dataloader
    zvdataloader.loadFromJsonFile(config_json_path)
    zvdataloader.test_subset.loadDataFromPaths(sameTperBatch=False)
    zvdataloader.createTestDataLoader(num_workers=8, pre_loading=True, shuffle=True,
                                      persistend_workers=True, pin_memory=True, pre_fetch_batch=2)


    if mode == "SB":
        return test_DSTCalib_SB(rtime_ctrller)
    elif mode in {"SB*", "PE", "PE+SB*"}:
        return test_DSTCalib_Joint(
            rtime_ctrller,
            min_iter_batch=config_loader.get("min_iter_batch", 60),
            mode=mode,
        )
    else:
        raise ValueError(
            f"Unsupported test mode: {mode}. "
            "Expected one of: SB, SB*, PE, PE+SB*."
        )


def test_byJsonList_testver(json_path_list, test_ver="DST_Calib", mode="SB"):
    for json_path in json_path_list:
        testAuto(test_ver, json_path, mode=mode)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Test DST-Calib from a JSON runtime config."
    )
    parser.add_argument(
        "--test-ver",
        default=None,
        help="Test version name (default: DST_Calib).",
    )
    parser.add_argument(
        "--config",
        dest="config_json_path",
        default=None,
        help="Path to the runtime JSON config.",
    )
    parser.add_argument(
        "--mode",
        choices=("SB", "SB*", "PE", "PE+SB*"),
        default=None,
        help="Testing mode (default: SB).",
    )
    parser.add_argument("legacy_args", nargs="*", help=argparse.SUPPRESS)
    parsed_args = parser.parse_args(args)

    if parsed_args.legacy_args:
        if any(
            value is not None
            for value in (
                parsed_args.test_ver,
                parsed_args.config_json_path,
                parsed_args.mode,
            )
        ):
            parser.error(
                "optional arguments cannot be combined with positional arguments"
            )

        if len(parsed_args.legacy_args) == 2:
            parsed_args.test_ver = "DST_Calib"
            parsed_args.config_json_path, parsed_args.mode = parsed_args.legacy_args
        elif len(parsed_args.legacy_args) == 3:
            (
                parsed_args.test_ver,
                parsed_args.config_json_path,
                parsed_args.mode,
            ) = parsed_args.legacy_args
        else:
            parser.error(
                "expected CONFIG MODE or TEST_VER CONFIG MODE positional arguments"
            )

        if parsed_args.mode not in {"SB", "SB*", "PE", "PE+SB*"}:
            parser.error(
                "mode must be one of: SB, SB*, PE, PE+SB*"
            )
    else:
        parsed_args.test_ver = parsed_args.test_ver or "DST_Calib"
        parsed_args.mode = parsed_args.mode or "SB"

    return parsed_args


def TestMain():
    while True:
        print("----------------------------------")
        print("1-> Test DST-Calib using environment variables")
        print("2-> Test DST-Calib using hardcoded JSON paths")
        print("q-> exit")
        order = input()
        if order == "1":
            config_json_path = os.environ.get("DSTCalib_CONFIG_JSON")
            test_version = os.environ.get("DSTCalib_TEST_VERSION")
            test_mode = os.environ.get("DSTCalib_TEST_MODE", "SB")

            if not config_json_path or not test_version:
                raise RuntimeError(
                    "order 1 requires DSTCalib_CONFIG_JSON and "
                    "DSTCalib_TEST_VERSION"
                )

            testAuto(test_version, config_json_path, mode=test_mode)
        elif order == "2":
            # Run the configured batch evaluation.
            json_path_list = [
                r"/path/to/test.json"   # place your json path
            ]
            test_byJsonList_testver(json_path_list, "DST_Calib", mode = "SB")
        
        elif order == "q":
            return
        else:
            print("Input error\n")


if __name__ == "__main__":
    args = parse_args()
    if args.config_json_path:
        testAuto(args.test_ver, args.config_json_path, mode=args.mode)
    else:
        TestMain()
