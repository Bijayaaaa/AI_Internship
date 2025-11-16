import requests
import time
import concurrent.futures
import statistics

# Use full path for your sample image
IMAGE_PATH = r"D:\AI internship\ai_intern\densenet-fastapi-app\test_images\sample.jpg"
URL = "http://127.0.0.1:8080/predict"

def send_request():
    try:
        with open(IMAGE_PATH, "rb") as f:
            files = {"file": ("sample.jpg", f, "image/jpeg")}
            start = time.time()
            response = requests.post(URL, files=files, timeout=15)
            end = time.time()
            latency = end - start
            if response.status_code == 200:
                return latency
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def benchmark(n_requests=50, concurrent=5):
    print("Starting benchmark...")
    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(send_request) for _ in range(n_requests)]
        for future in concurrent.futures.as_completed(futures):
            t = future.result()
            if t:
                times.append(t)
    print(f"\n✅ Total successful requests: {len(times)}")
    if times:
        print(f"📉 Average latency: {statistics.mean(times)*1000:.2f} ms")
        print(f"📈 95th percentile latency: {statistics.quantiles(times, n=100)[94]*1000:.2f} ms")
        print(f"⚡ Throughput: {len(times)/sum(times):.2f} req/sec")

if __name__ == "__main__":
    benchmark(n_requests=50, concurrent=5)
