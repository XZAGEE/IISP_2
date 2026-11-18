
import argparse
import json
import time
import urllib.request
import urllib.error
import statistics
import csv
import os


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def send_request(url, audio_path):
    boundary = "----BenchmarkBoundary123456"

    with open(audio_path, "rb") as file:
        audio_data = file.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_data + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        url + "/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json"
        },
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode())

    client_latency = (time.perf_counter() - start) * 1000

    return result, client_latency


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default="samples/test_yes.wav"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=30
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    args = parser.parse_args()

    containers = {
        "multi": "http://localhost:8000",
        "single": "http://localhost:8001"
    }

    results = {
        name: {
            "client": [],
            "server": []
        }
        for name in containers
    }

    print("Перевірка доступності API...")

    for name, url in containers.items():
        print(f"{name}: {url}")

    print("\nПрогрів моделей...")

    for name, url in containers.items():
        for _ in range(args.warmup):
            send_request(url, args.file)

    print("Прогрів завершено.")
    print(f"\nВиконується {args.iterations} вимірювань...")

    for i in range(args.iterations):
        # Чергуємо порядок запитів для зменшення впливу порядку
        order = list(containers.items())

        if i % 2 == 1:
            order.reverse()

        for name, url in order:
            response, client_time = send_request(url, args.file)

            results[name]["client"].append(client_time)
            results[name]["server"].append(response["latency_ms"])

            if response["predicted_class"] != "yes":
                print(
                    f"Попередження: {name} передбачив "
                    f"{response['predicted_class']}"
                )

        print(f"Вимірювання {i + 1}/{args.iterations}")

    print("\n========== РЕЗУЛЬТАТИ ==========")

    rows = []

    for name, measurements in results.items():
        print(f"\n{name.upper()}")

        row = {"container": name}

        for metric, values in measurements.items():
            mean = statistics.mean(values)
            p50 = percentile(values, 50)
            p95 = percentile(values, 95)

            print(f"{metric}:")
            print(f"  Mean: {mean:.2f} ms")
            print(f"  P50:  {p50:.2f} ms")
            print(f"  P95:  {p95:.2f} ms")

            row[f"{metric}_mean"] = round(mean, 2)
            row[f"{metric}_p50"] = round(p50, 2)
            row[f"{metric}_p95"] = round(p95, 2)

        rows.append(row)

    os.makedirs("results", exist_ok=True)

    with open(
        "results/benchmark.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nРезультати збережено у results/benchmark.csv")


if __name__ == "__main__":
    main()
