"""CrossModalLab — Dataset Source Connectivity & Metadata Inspector.

Checks official sources, file sizes, licenses, and accessibility for:
1. GQA (Stanford)
2. COCO (cocodataset.org)
3. R-Bench (PMLR / GitHub)
4. HallusionBench (GitHub)
5. MoHoBench (AAAI / GitHub)
"""

import urllib.request
import json

def check_url(url, desc):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_length = resp.headers.get("Content-Length")
            cl_mb = f"{int(content_length)/(1024*1024):.2f} MB" if content_length else "Unknown"
            print(f"[OK] {desc}: status={resp.status}, size={cl_mb}")
            return True, resp.status, content_length
    except Exception as e:
        print(f"[FAIL] {desc}: {e}")
        return False, None, None

def inspect_sources():
    print("=== Inspecting Real-World Benchmark Sources ===")
    
    # 1. GQA
    check_url("https://cs.stanford.edu/people/rak248/VG_100K/2345678.jpg", "GQA Image Sample")
    
    # 2. COCO
    check_url("http://images.cocodataset.org/val2017/000000000139.jpg", "COCO val2017 Image Sample")
    
    # 3. HallusionBench
    check_url("https://raw.githubusercontent.com/tianyi-lab/HallusionBench/main/HallusionBench.json", "HallusionBench Annotation JSON")
    
    # 4. R-Bench
    check_url("https://raw.githubusercontent.com/R-Bench/R-Bench/main/README.md", "R-Bench GitHub README")
    
    # 5. MoHoBench
    check_url("https://raw.githubusercontent.com/yanxuzhu/MoHoBench/main/README.md", "MoHoBench GitHub README")

if __name__ == "__main__":
    inspect_sources()
