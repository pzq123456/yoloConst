# launcher.py — 崩溃自动重启看门狗
"""
用法:
    uv run python src/launcher.py src/main.py
    uv run python src/launcher.py src/monitor.py

子进程因 ffmpeg 断言等 C 层崩溃退出时，自动等待 5s 重新拉起。
"""
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("用法: python launcher.py <脚本路径>")
        sys.exit(1)

    target = sys.argv[1]
    cwd = Path(target).parent

    restart_count = 0

    while True:
        print(f"\n{'='*50}")
        print(f"[WATCHDOG] 启动: {target}")
        if restart_count:
            print(f"[WATCHDOG] 已重启 {restart_count} 次")
        print(f"{'='*50}\n")

        proc = subprocess.Popen(
            ["uv", "run", "python", target],
            cwd=str(cwd),
        )
        exit_code = proc.wait()

        if exit_code == 0:
            print(f"[WATCHDOG] 正常退出 (exit=0)")
            break

        # 非零退出 = C 层崩溃（ffmpeg assertion 等）
        restart_count += 1
        print(f"[WATCHDOG] 异常退出 (exit={exit_code})，5s 后重启...")
        time.sleep(5)


if __name__ == "__main__":
    main()
