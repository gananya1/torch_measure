import modal

app = modal.App("ncf-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "sentence-transformers", "datasets",
        "huggingface_hub", "scikit-learn", "scipy", "tqdm",
        "pandas", "pyarrow",
    )
    .pip_install("git+https://github.com/gananya1/torch_measure.git@competition")
    .add_local_dir("competition", remote_path="/competition")
)

volume = modal.Volume.from_name("ncf-outputs", create_if_missing=True)

@app.function(image=image, gpu="A10G", timeout=7200,
              volumes={"/outputs": volume})
def train():
    import subprocess, shutil

    subprocess.run([
        "python", "/competition/train_ncf.py",
        "--encoder",               "all-MiniLM-L6-v2",
        "--embed-dim",             "384",
        "--epochs",                "10",
        "--output",                "/outputs/ncf_head.pt",
        "--subject-cache-output",  "/outputs/subject_cache.pkl",
        "--embeddings-checkpoint", "/outputs/ncf_embeddings.pt",
    ], check=True)

    subprocess.run(["python", "/competition/fit_centroids.py"], check=True)
    shutil.copy("/competition/centroids.npy", "/outputs/centroids.npy")

@app.local_entrypoint()
def main():
    train.remote()
    vol = modal.Volume.from_name("ncf-outputs")
    for name in ("ncf_head.pt", "subject_cache.pkl", "centroids.npy"):
        data = vol.read_file(name)
        with open(f"competition/{name}", "wb") as f:
            f.write(data)
    print("Downloaded artifacts to competition/")