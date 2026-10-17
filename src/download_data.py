"""Download the exact public source release. No API key is required."""
from pathlib import Path
import hashlib, json, urllib.request, shutil
ROOT=Path(__file__).resolve().parents[1]
SHA='061870127481dcd73c29df85ebcfddeac2ed0828'
PATH='consensus/2016_04_01_a549_48hr_batch1/2016_04_01_a549_48hr_batch1_consensus_median_dmso.csv.gz'
FILES={
 'data/raw/lincs_consensus_median_dmso.csv.gz':f'https://media.githubusercontent.com/media/broadinstitute/lincs-cell-painting/{SHA}/{PATH}',
 'data/raw/repurposing_info.tsv':f'https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/{SHA}/metadata/moa/repurposing_info.tsv',
 'references/utils_consensus_blocklist.txt':f'https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/{SHA}/utils/consensus_blocklist.txt',
}
def main():
    previous=ROOT/'data/source_manifest.json'
    expected={x['path']:x['sha256'] for x in json.loads(previous.read_text(encoding='utf-8'))} if previous.exists() else {}
    manifest=[]
    for name,url in FILES.items():
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            print('Downloading',name,flush=True)
            temp=target.with_suffix(target.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=120) as r,temp.open('wb') as f: shutil.copyfileobj(r,f)
            temp.replace(target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if name in expected and expected[name]!=digest: raise ValueError(f'Checksum mismatch: {name}')
        manifest.append({'path':name,'url':url,'source_commit':SHA,'bytes':target.stat().st_size,'sha256':digest,'license':'CC0-1.0 for source data'})
        print('Verified',name,digest,flush=True)
    previous.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
