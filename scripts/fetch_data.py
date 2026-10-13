"""Download every raw input of the study into HECA_DIR and $WORK/ext, skip files that are present with the expected
size, and check SHA-256 against inputs/data_manifest.json. With --write-manifest, record sizes and checksums of the
local files instead. Usage: fetch_data.py [--write-manifest] [--no-checksum]"""
import argparse, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
HECA = os.environ.get("HECA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "heca"))
RCA = "https://cellgeni.cog.sanger.ac.uk/vento/reproductivecellatlas"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
FILES = [
    (os.path.join(HECA, "endometriumAtlasV2_nuclei.h5ad"), f"{RCA}/endometriumAtlasV2_nuclei.h5ad"),
    (os.path.join(HECA, "endometriumAtlasV2_cells_with_counts.h5ad"), f"{RCA}/endometriumAtlasV2_cells_with_counts.h5ad"),
    (os.path.join(W, "ext/huang/huang2026.h5ad"), "https://datasets.cellxgene.cziscience.com/cf7d5037-4cda-42ef-8375-19f1f3f9deea.h5ad"),
    (os.path.join(W, "ext/bulk/GSE60129_series_matrix.txt.gz"), f"{GEO}/series/GSE60nnn/GSE60129/matrix/GSE60129_series_matrix.txt.gz"),
    (os.path.join(W, "ext/bulk/GSE137765_series_matrix.txt.gz"), f"{GEO}/series/GSE137nnn/GSE137765/matrix/GSE137765_series_matrix.txt.gz"),
    (os.path.join(W, "ext/bulk/GSE97929_series_matrix.txt.gz"), f"{GEO}/series/GSE97nnn/GSE97929/matrix/GSE97929_series_matrix.txt.gz"),
    (os.path.join(W, "ext/bulk/GSE97929_reads_all.txt.gz"), f"{GEO}/series/GSE97nnn/GSE97929/suppl/GSE97929_reads_all.txt.gz"),
    (os.path.join(W, "ext/bulk/GPL6244.annot.gz"), f"{GEO}/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz"),
    (os.path.join(W, "ext/natural/GSE4888_series_matrix.txt.gz"), f"{GEO}/series/GSE4nnn/GSE4888/matrix/GSE4888_series_matrix.txt.gz"),
    (os.path.join(W, "ext/natural/GSE6364_series_matrix.txt.gz"), f"{GEO}/series/GSE6nnn/GSE6364/matrix/GSE6364_series_matrix.txt.gz"),
    (os.path.join(W, "ext/natural/GPL570.annot.gz"), f"{GEO}/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"),
    (os.path.join(W, "ext/pgr/refGene_hg19.txt.gz"), "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"),
] + [(os.path.join(W, f"ext/pgr/{f}.bw"), f"{GEO}/samples/{f[:7]}nnn/{f.split('_')[0]}/suppl/{f}.bw")
     for f in ("GSM3890632_3_N252i8", "GSM3890633_4_N275i10", "GSM3890634_5_Inputi79", "GSM3890635_1_N293i93",
               "GSM3890636_2_N274i94", "GSM3890637_3_N293i95", "GSM3890638_4_N274i96")]
MANIFEST = os.path.join(ROOT, "inputs", "data_manifest.json")


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write-manifest", action="store_true"); ap.add_argument("--no-checksum", action="store_true"); a = ap.parse_args()
    key = lambda p: os.path.basename(p)
    if a.write_manifest:
        man = {key(p): dict(url=u, bytes=os.path.getsize(p), sha256=sha256(p)) for p, u in FILES}
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True); json.dump(man, open(MANIFEST, "w"), indent=1); print(f"wrote {MANIFEST}"); sys.exit(0)
    man = json.load(open(MANIFEST)); bad = []
    for p, u in FILES:
        m = man[key(p)]; os.makedirs(os.path.dirname(p), exist_ok=True)
        if not (os.path.exists(p) and os.path.getsize(p) == m["bytes"]):
            print(f"downloading {key(p)} ({m['bytes'] / 1e9:.2f} GB)", flush=True)
            subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "5", "-o", p, u], check=True)
        ok = os.path.getsize(p) == m["bytes"] and (a.no_checksum or sha256(p) == m["sha256"])
        print(f"{'ok ' if ok else 'BAD'} {key(p)}", flush=True); bad += [] if ok else [key(p)]
    if bad:
        sys.exit(f"checksum or size mismatch: {bad}")
