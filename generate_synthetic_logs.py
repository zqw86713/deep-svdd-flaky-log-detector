import os
import random
import datetime
import csv
from pathlib import Path

# 配置
NUM_NORMAL_LOGS = 150
NUM_FLAKY_LOGS = 50
OUTPUT_DIR = Path("data/raw_logs")

random.seed(42)


NORMAL_TEST_NAMES = [
    "test_login_success",
    "test_user_registration",
    "test_update_profile",
    "test_delete_account",
    "test_list_orders",
    "test_create_order",
    "test_payment_success",
    "test_logout",
]

FLAKY_REASONS = [
    "timeout",
    "network",
    "random",
    "order_dependent",
    "external_service",
]

PYTEST_HEADERS = [
    "============================= test session starts ==============================",
    "platform linux -- Python 3.10.0, pytest-7.0.0",
    "cachedir: .pytest_cache",
    "rootdir: /workspace/project",
    "plugins: cov-3.0.0, xdist-2.4.0",
]

def random_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(
        minutes=random.randint(-60, 0),
        seconds=random.randint(0, 59)
    )
    return (now + delta).strftime("%Y-%m-%d %H:%M:%S")


def gen_normal_log_content():
    lines = []
    lines.extend(PYTEST_HEADERS)
    lines.append("collected {} items".format(random.randint(5, 20)))
    lines.append("")

    num_tests = random.randint(5, 20)
    for i in range(num_tests):
        name = random.choice(NORMAL_TEST_NAMES)
        duration = round(random.uniform(0.01, 0.5), 2)
        lines.append("tests/test_app.py::{} PASSED [{:>3}%]".format(
            name, int((i + 1) / num_tests * 100)
        ))
        lines.append("")

    lines.append("============================== {} passed in {:.2f}s ===============================".format(
        num_tests, random.uniform(1.0, 5.0)
    ))
    return "\n".join(lines)


def gen_flaky_log_content(flaky_reason: str):
    lines = []
    lines.extend(PYTEST_HEADERS)
    lines.append("collected {} items".format(random.randint(5, 20)))
    lines.append("")

    num_tests = random.randint(5, 20)
    fail_index = random.randint(0, num_tests - 1)

    for i in range(num_tests):
        name = random.choice(NORMAL_TEST_NAMES)
        duration = round(random.uniform(0.01, 1.5), 2)

        if i == fail_index:
            # 这里制造 flaky-like 行为
            if flaky_reason == "timeout":
                lines.append(f"tests/test_app.py::{name} FAILED [{int((i + 1) / num_tests * 100):>3}%]")
                lines.append("")
                lines.append("___________________________________ ERROR ___________________________________")
                lines.append(f"tests/test_app.py::{name}")
                lines.append(f"{random_timestamp()} ERROR    TimeoutError: test exceeded timeout of {round(random.uniform(1.0, 5.0), 1)}s")
                lines.append("E   TimeoutError: Operation timed out")
                lines.append("")
            elif flaky_reason == "network":
                lines.append(f"tests/test_app.py::{name} FAILED [{int((i + 1) / num_tests * 100):>3}%]")
                lines.append("")
                lines.append("___________________________________ ERROR ___________________________________")
                lines.append(f"tests/test_app.py::{name}")
                lines.append(f"{random_timestamp()} ERROR    ConnectionError: failed to reach api.example.com")
                lines.append("E   requests.exceptions.ConnectionError: Max retries exceeded with url")
                lines.append("")
            elif flaky_reason == "random":
                # 有时候 pass 有时候 fail 的感觉：比如 AssertionError 带随机数
                if random.random() < 0.5:
                    lines.append(f"tests/test_app.py::{name} PASSED [{int((i + 1) / num_tests * 100):>3}%]")
                    lines.append("")
                else:
                    lines.append(f"tests/test_app.py::{name} FAILED [{int((i + 1) / num_tests * 100):>3}%]")
                    lines.append("")
                    lines.append("___________________________________ FAILURES ___________________________________")
                    lines.append(f"___________________________ {name} ___________________________")
                    lines.append("")
                    lines.append(f"    assert {random.randint(0, 10)} == {random.randint(0, 10)}")
                    lines.append("E   AssertionError: assert values differ (random seed issue?)")
                    lines.append("")
            elif flaky_reason == "order_dependent":
                lines.append(f"tests/test_app.py::{name} FAILED [{int((i + 1) / num_tests * 100):>3}%]")
                lines.append("")
                lines.append("___________________________________ FAILURES ___________________________________")
                lines.append(f"___________________________ {name} ___________________________")
                lines.append("")
                lines.append("E   AssertionError: expected user to be logged out, but session is still active")
                lines.append("E   Note: test may depend on execution order or shared state")
                lines.append("")
            elif flaky_reason == "external_service":
                lines.append(f"tests/test_app.py::{name} FAILED [{int((i + 1) / num_tests * 100):>3}%]")
                lines.append("")
                lines.append("___________________________________ ERROR ___________________________________")
                lines.append(f"tests/test_app.py::{name}")
                lines.append("E   ExternalServiceError: 503 Service Unavailable")
                lines.append("E   ValueError: Could not fetch configuration from remote service")
                lines.append("")
        else:
            lines.append("tests/test_app.py::{} PASSED [{:>3}%]".format(
                name, int((i + 1) / num_tests * 100)
            ))
            lines.append("")

    # pytest summary 行
    lines.append("======================== short test summary info ========================")
    lines.append("FAILED tests/test_app.py::{} - flaky reason: {}".format(
        name, flaky_reason
    ))
    lines.append("!!!!!!!!!!!!!!!!!!!! xdist.dsession.Interrupted !!!!!!!!!!!!!!!!!!!!")
    lines.append("1 failed, {} passed in {:.2f}s".format(
        num_tests - 1, random.uniform(3.0, 15.0)
    ))
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = OUTPUT_DIR.parent / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "reason"])

        # 生成正常日志
        for i in range(NUM_NORMAL_LOGS):
            fname = f"normal_{i:03d}.log"
            content = gen_normal_log_content()
            (OUTPUT_DIR / fname).write_text(content, encoding="utf-8")
            writer.writerow([fname, 0, "normal"])

        # 生成 flaky-like 日志
        for i in range(NUM_FLAKY_LOGS):
            reason = random.choice(FLAKY_REASONS)
            fname = f"flaky_{reason}_{i:03d}.log"
            content = gen_flaky_log_content(reason)
            (OUTPUT_DIR / fname).write_text(content, encoding="utf-8")
            writer.writerow([fname, 1, reason])

    print(f"Generated {NUM_NORMAL_LOGS} normal logs and {NUM_FLAKY_LOGS} flaky-like logs.")
    print(f"Logs saved to: {OUTPUT_DIR}")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
